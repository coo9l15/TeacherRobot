import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import time
from collections import deque
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication
import sys
import mediapipe as mp

class CalibrationStage:
    def __init__(self, name, instruction, duration=3.0, position="center"):
        self.name = name
        self.instruction = instruction
        self.duration = duration
        self.position = position  # center, left, right, up, down
        self.data = []

class EyeTracker:
    def __init__(self, smoothing_window=10, lightweight_mode=True):  # Added lightweight mode option
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduced resolution for lightweight mode
        self.process_width = 240 if lightweight_mode else 320
        self.process_height = 180 if lightweight_mode else 240
        
        self.lightweight_mode = lightweight_mode
        
        self.detector = FaceMeshDetector(
            maxFaces=1,
            minDetectionCon=0.5 if lightweight_mode else 0.7,  # Lower detection threshold for better performance
            minTrackCon=0.5 if lightweight_mode else 0.7       # Lower tracking confidence for better performance
        )
        
        self.smoothing_window = smoothing_window
        self.ear_history = deque(maxlen=smoothing_window)
        self.ear_min_history = deque(maxlen=30)  # Track minimum EARs for adaptive thresholding
        self.ear_max_history = deque(maxlen=30)  # Track maximum EARs for adaptive thresholding
        
        # Default thresholds, will be calibrated
        self.ear_threshold = 0.22 # Default threshold
        self.head_pose_threshold = 35
        self.adaptive_threshold_enabled = True  # Enable adaptive thresholding
        
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        self.processing_thread = None
        self.is_running = True
        self.latest_result = False
        self.lock = threading.Lock()
        
        self.user_present = False
        self.no_user_frames = 0
        self.no_user_threshold = 10
        
        # Student identification data
        self.current_student_id = None
        self.student_ear_profiles = {}
        
        # Status callback for UI updates
        self.status_callback = None
        self.progress_callback = None
        
        # Gaze-based attention tracking
        self.gaze_history = deque(maxlen=smoothing_window)
        
        # Initialize MediaPipe FaceMesh with iris tracking only if not in lightweight mode
        if not self.lightweight_mode:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,  # Enables iris landmarks
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe Face Mesh: {str(e)}")
                self.face_mesh = None
                self.lightweight_mode = True
        else:
            self.face_mesh = None
        
        # Process skipping for lightweight mode (process every N frames)
        self.frame_skip = 1  # Default: process every frame
        if self.lightweight_mode:
            self.frame_skip = 2  # Process every other frame
            
        self.skip_counter = 0
        
        # Start processing immediately by default
        self.start_processing()
        
    def start_processing(self):
        """Start background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
        
    def _process_frames(self):
        """Background thread for continuous frame processing"""
        while self.is_running:
            success, img = self.cap.read()
            if not success:
                time.sleep(0.01) # Brief pause if cap.read() fails
                continue
            
            # Skip frames for performance if in lightweight mode
            self.skip_counter = (self.skip_counter + 1) % self.frame_skip
            if self.skip_counter != 0 and self.frame_skip > 1:
                time.sleep(0.001)  # Just update counter but don't process
                continue
                
            # Resize with lower resolution in lightweight mode
            img = cv2.resize(img, (self.process_width, self.process_height))
            
            # In lightweight mode, skip the expensive MediaPipe processing
            if self.lightweight_mode:
                img, faces = self.detector.findFaceMesh(img, draw=False)
                
                with self.lock:
                    if not faces:
                        self.no_user_frames += 1
                        if self.no_user_frames >= self.no_user_threshold:
                            self.user_present = False
                            self.latest_result = False
                    else:
                        self.no_user_frames = 0
                        self.user_present = True
                        
                        # Use the traditional EAR method for lightweight processing
                        face = faces[0]
                        try:
                            left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
                            right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
                            
                            # Calculate EAR with the enhanced method
                            left_ear = self._calculate_ear_enhanced(left_eye)
                            right_ear = self._calculate_ear_enhanced(right_eye)
                            
                            # Simple head pose check - avoid expensive 3D projection
                            head_in_position = True  # Simplified for performance
                            
                            # Track min/max EAR values for adaptive thresholding
                            avg_ear = (left_ear + right_ear) / 2
                            if avg_ear > 0.1:  # Only track reasonable values
                                self.ear_max_history.append(avg_ear)
                                if avg_ear < 0.3:  # Likely not fully open
                                    self.ear_min_history.append(avg_ear)
                            
                            # Adaptive threshold calculation if enabled and we have enough data
                            if self.adaptive_threshold_enabled and len(self.ear_min_history) > 5 and len(self.ear_max_history) > 5:
                                min_ear = np.percentile(list(self.ear_min_history), 10)  # 10th percentile of minimums
                                max_ear = np.percentile(list(self.ear_max_history), 90)  # 90th percentile of maximums
                                
                                # Only update if we have a meaningful difference
                                if max_ear - min_ear > 0.05:
                                    # Set threshold at 35% between min and max (closer to min) - more permissive
                                    adaptive_threshold = min_ear + (max_ear - min_ear) * 0.35
                                    
                                    # Ensure the adaptive threshold is within reasonable bounds
                                    adaptive_threshold = max(0.1, min(0.3, adaptive_threshold))
                                    
                                    # Gradually adjust the main threshold (with stronger adaptation)
                                    self.ear_threshold = self.ear_threshold * 0.9 + adaptive_threshold * 0.1
                            
                            if head_in_position:
                                # Calculate weighted EAR (more weight to the lower eye)
                                weighted_ear = min(left_ear, right_ear) * 0.6 + max(left_ear, right_ear) * 0.4
                                
                                # Add to history
                                self.ear_history.append(weighted_ear)
                                
                                # Calculate smoothed EAR - simplified for performance
                                if self.ear_history:
                                    # Use mean instead of percentile for better performance
                                    smoothed_ear = sum(self.ear_history) / len(self.ear_history)
                                else:
                                    smoothed_ear = weighted_ear
                                
                                # Use a permissive threshold 
                                self.latest_result = smoothed_ear > (self.ear_threshold * 0.9)
                            else:
                                self.latest_result = False
                        except Exception as e:
                            # If there was an error in processing, assume not looking
                            self.latest_result = False
            else:
                # Full processing with MediaPipe for better accuracy
                img, faces = self.detector.findFaceMesh(img, draw=False)
                
                with self.lock:
                    if not faces:
                        self.no_user_frames += 1
                        if self.no_user_frames >= self.no_user_threshold:
                            self.user_present = False
                            self.latest_result = False
                        # Continue to update FPS even if no face
                    else:
                        self.no_user_frames = 0
                        self.user_present = True
                        
                        # Only process MediaPipe if face mesh is available
                        if self.face_mesh:
                            # Convert to RGB and process
                            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            
                            if results.multi_face_landmarks:
                                # Extract landmarks and determine if looking at camera
                                landmarks = results.multi_face_landmarks[0]
                                self.latest_result = self.is_looking_at_camera(landmarks)
                            else:
                                # Fall back to traditional method if MediaPipe doesn't detect landmarks
                                self._process_traditional_method(faces[0])
                        else:
                            # If face_mesh not available, use traditional method
                            self._process_traditional_method(faces[0])
            
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            
            time.sleep(0.001)
    
    def _process_traditional_method(self, face):
        """Process using the traditional EAR method as fallback"""
        try:
            left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
            right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
            
            # Calculate EAR with the enhanced method
            left_ear = self._calculate_ear_enhanced(left_eye)
            right_ear = self._calculate_ear_enhanced(right_eye)
            
            # Head pose check - only if not in lightweight mode
            if not self.lightweight_mode:
                pitch, yaw, _ = self._calculate_head_pose(face)
                head_in_position = pitch is None or yaw is None or (abs(pitch) <= self.head_pose_threshold and abs(yaw) <= self.head_pose_threshold)
            else:
                head_in_position = True  # Skip head pose in lightweight mode
            
            # Update adaptive thresholds
            avg_ear = (left_ear + right_ear) / 2
            if avg_ear > 0.1:
                self.ear_max_history.append(avg_ear)
                if avg_ear < 0.3:
                    self.ear_min_history.append(avg_ear)
            
            if not head_in_position:
                self.latest_result = False
            else:
                # Calculate weighted EAR
                weighted_ear = min(left_ear, right_ear) * 0.6 + max(left_ear, right_ear) * 0.4
                
                # Add to history
                self.ear_history.append(weighted_ear)
                
                # Calculate smoothed EAR - use mean if in lightweight mode for better performance
                if self.ear_history:
                    if self.lightweight_mode:
                        smoothed_ear = sum(self.ear_history) / len(self.ear_history)
                    else:
                        smoothed_ear = np.percentile(list(self.ear_history), 30)
                else:
                    smoothed_ear = weighted_ear
                
                # Use a permissive threshold
                self.latest_result = smoothed_ear > (self.ear_threshold * 0.9)
        except Exception as e:
            # If there was an error in processing, assume not looking
            self.latest_result = False

    def is_looking_at_camera(self, landmarks=None):
        """Check if the user is looking at the camera"""
        with self.lock:
            # If called directly without landmarks, return the cached result
            if landmarks is None:
                return self.latest_result if self.user_present else False
                
            # If landmarks are provided, process them to determine attention
            try:
                # Get iris landmarks (468-478 are for left and right irises)
                left_iris = []
                right_iris = []
                left_eye_landmarks = []
                right_eye_landmarks = []
                
                # Get iris and eye landmarks
                for i, landmark in enumerate(landmarks.landmark):
                    # Left iris points
                    if 468 <= i <= 472:
                        left_iris.append((landmark.x, landmark.y))
                    # Right iris points
                    elif 473 <= i <= 477:
                        right_iris.append((landmark.x, landmark.y))
                    # Left eye landmarks
                    elif i in [33, 133, 157, 158, 159, 160, 161, 173, 246]:
                        left_eye_landmarks.append((landmark.x, landmark.y))
                    # Right eye landmarks
                    elif i in [263, 362, 385, 386, 387, 388, 389, 390, 466]:
                        right_eye_landmarks.append((landmark.x, landmark.y))
                
                # If we don't have enough landmarks, return False
                if not left_iris or not right_iris or not left_eye_landmarks or not right_eye_landmarks:
                    return False
                
                # Calculate eye centers
                left_eye_center_x = sum(x for x, _ in left_eye_landmarks) / len(left_eye_landmarks)
                left_eye_center_y = sum(y for _, y in left_eye_landmarks) / len(left_eye_landmarks)
                right_eye_center_x = sum(x for x, _ in right_eye_landmarks) / len(right_eye_landmarks)
                right_eye_center_y = sum(y for _, y in right_eye_landmarks) / len(right_eye_landmarks)
                
                # Calculate iris centers
                left_iris_center_x = sum(x for x, _ in left_iris) / len(left_iris)
                left_iris_center_y = sum(y for _, y in left_iris) / len(left_iris)
                right_iris_center_x = sum(x for x, _ in right_iris) / len(right_iris)
                right_iris_center_y = sum(y for _, y in right_iris) / len(right_iris)
                
                # Calculate left and right eye width (horizontal distance)
                left_eye_width = max(x for x, _ in left_eye_landmarks) - min(x for x, _ in left_eye_landmarks)
                right_eye_width = max(x for x, _ in right_eye_landmarks) - min(x for x, _ in right_eye_landmarks)
                
                # Calculate left and right eye height (vertical distance)
                left_eye_height = max(y for _, y in left_eye_landmarks) - min(y for _, y in left_eye_landmarks)
                right_eye_height = max(y for _, y in right_eye_landmarks) - min(y for _, y in right_eye_landmarks)
                
                # Calculate the relative position of the iris within the eye
                # Normalized to [-1, 1] where 0 is center
                left_iris_rel_x = (left_iris_center_x - left_eye_center_x) / (left_eye_width / 2) if left_eye_width > 0 else 0
                left_iris_rel_y = (left_iris_center_y - left_eye_center_y) / (left_eye_height / 2) if left_eye_height > 0 else 0
                right_iris_rel_x = (right_iris_center_x - right_eye_center_x) / (right_eye_width / 2) if right_eye_width > 0 else 0
                right_iris_rel_y = (right_iris_center_y - right_eye_center_y) / (right_eye_height / 2) if right_eye_height > 0 else 0
                
                # Calculate the distance of iris from center
                left_iris_distance = np.sqrt(left_iris_rel_x**2 + left_iris_rel_y**2)
                right_iris_distance = np.sqrt(right_iris_rel_x**2 + right_iris_rel_y**2)
                
                # Threshold for determining if looking at camera
                # Lower value = more strict, higher value = more lenient
                threshold = 0.8
                
                # Consider looking if both irises are relatively centered
                is_looking = (left_iris_distance < threshold and right_iris_distance < threshold)
                
                # Store additional info in gaze history for smoothing
                avg_distance = (left_iris_distance + right_iris_distance) / 2
                self.gaze_history.append(1.0 - min(1.0, avg_distance))
                
                # Calculate smoothed gaze attention using history
                if len(self.gaze_history) > 0:
                    # More weight to recent values for faster response
                    weights = np.linspace(0.5, 1.0, len(self.gaze_history))
                    weighted_sum = sum(w * g for w, g in zip(weights, self.gaze_history))
                    weighted_avg = weighted_sum / sum(weights)
                    
                    # Final determination with threshold of 0.6 (more permissive)
                    return weighted_avg > 0.6
                
                return is_looking
                
            except Exception as e:
                # If there's any error, assume not looking
                return False

    def is_user_present(self):
        """Check if a user is present in front of the camera"""
        with self.lock:
            return self.user_present

    def _calculate_head_pose(self, face):
        """Optimized head pose calculation"""
        try:
            # Check if we have enough points in the face
            if len(face) < 468:  # Standard mediapipe face mesh has 468 points
                return None, None, None
                
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye
                (225.0, 170.0, -135.0),   # Right eye
            ])
            
            # Ensure all required indices exist
            if 1 not in face or 152 not in face or 33 not in face or 263 not in face:
                return None, None, None
                
            image_points = np.array([
                face[1], face[152], face[33], face[263],
            ], dtype="double")
            
            # Check for invalid points
            if np.isnan(image_points).any() or not np.all(np.isfinite(image_points)):
                return None, None, None
                
            camera_matrix = np.array(
                [[640, 0, 160],[0, 640, 120],[0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4,1))
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
                
            if not success:
                return None, None, None
                
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            return euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]
        except Exception:
            return None, None, None

    def _calculate_ear_enhanced(self, eye):
        """Enhanced EAR calculation with more emphasis on vertical opening"""
        try:
            # Safety check for eye points
            if not eye or len(eye) != 6:
                return 0
                
            eye = np.array(eye)
            
            # Check for NaN or infinite values
            if np.isnan(eye).any() or not np.all(np.isfinite(eye)):
                return 0
                
            # Calculate horizontal distance (width)
            C = np.linalg.norm(eye[0] - eye[3])
            
            # Calculate vertical distances (height) at multiple points for robustness
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            
            # Additional vertical measurement for better accuracy
            midpoint1 = (eye[1] + eye[2]) / 2
            midpoint2 = (eye[4] + eye[5]) / 2
            D = np.linalg.norm(midpoint1 - midpoint2)
            
            # Avoid division by zero
            if C == 0:
                return 0
                
            # Enhanced EAR calculation with more weight on vertical opening
            # and using multiple vertical measurements for robustness
            ear = ((A + B) * 0.5 + D * 0.5) / (C * 1.0)
            
            # Normalize to a reasonable range
            return ear if 0 < ear < 1 else 0
        except Exception:
            return 0
            
    def get_fps(self):
        """Get current processing FPS"""
        return self.fps

    def update_status(self, message):
        """Update status message (can be connected to UI)"""
        print(message)  # Default to console output
        if self.status_callback:
            self.status_callback(message)
            
    def set_status_callback(self, callback):
        """Set callback function for status updates"""
        self.status_callback = callback

    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback

    def release(self):
        """Clean up resources"""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def start_calibration(self):
        """Start the calibration process in a separate thread"""
        # Pause the main processing thread during calibration
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.processing_thread = None
        
        # Create and start the calibration thread
        self.update_status("Starting interactive calibration...")
        self.calib_worker = CalibrationWorker(self)
        self.calib_thread = QThread()
        self.calib_worker.moveToThread(self.calib_thread)
        self.calib_thread.started.connect(self.calib_worker.run)
        self.calib_worker.progress_update.connect(self.update_progress)
        self.calib_worker.status_update.connect(self.update_status)
        self.calib_worker.finished.connect(self.on_calibration_done)
        self.calib_worker.finished.connect(self.calib_thread.quit)
        self.calib_worker.finished.connect(self.calib_worker.deleteLater)
        self.calib_thread.finished.connect(self.calib_thread.deleteLater)
        self.calib_thread.start()

    def update_progress(self, progress):
        """Update calibration progress"""
        if self.progress_callback:
            self.progress_callback(progress)

    def on_calibration_done(self, threshold):
        """Handle completion of calibration"""
        self.ear_threshold = threshold
        self.is_running = True  # Make sure processing can run
        self.start_processing()  # Restart the processing thread
        self.update_status(f'Calibration done. EAR threshold: {threshold:.3f}')

    def calibrate(self):
        """Non-Qt calibration method that works without QApplication"""
        self.update_status("Starting interactive calibration...")
        threshold = self._calibrate_interactive()
        self.ear_threshold = threshold
        self.is_running = True
        self.start_processing()
        self.update_status(f'Calibration done. EAR threshold: {threshold:.3f}')
        return threshold

    def calibrate_for_student(self, student_id, student_name=""):
        """Calibrate for a specific student and store their profile"""
        # Make sure we're not already calibrating
        if hasattr(self, '_calibrating') and self._calibrating:
            self.update_status("Calibration already in progress")
            return 0.22  # default threshold
            
        self._calibrating = True
        
        try:
            # Stop the processing thread temporarily
            was_running = self.is_running
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            # Ensure camera is accessible by releasing and reopening it
            if self.cap.isOpened():
                self.cap.release()
                
            # Wait a moment for the camera to be released fully
            time.sleep(0.2)
            
            # Reopen the camera
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Make sure camera opened successfully
            if not self.cap.isOpened():
                self.update_status("Could not open camera for calibration")
                return 0.22  # default threshold
            
            # Run calibration
            self.update_status(f"Calibrating for student: {student_name or student_id}")
            threshold = self._calibrate_interactive()
            
            # Store student profile
            self.student_ear_profiles[student_id] = {
                'threshold': threshold,
                'name': student_name,
                'calibration_time': time.time()
            }
            
            self.ear_threshold = threshold  # Set as current threshold
            self.current_student_id = student_id
            
            return threshold
        except Exception as e:
            self.update_status(f"Calibration error: {str(e)}")
            return 0.22  # default threshold
        finally:
            self._calibrating = False
            # Make sure all windows are closed
            try:
                cv2.destroyAllWindows()
                for i in range(5):
                    cv2.waitKey(1)
            except:
                pass
                
            # Restart processing thread if it was running before
            if was_running:
                self.is_running = True
                self.start_processing()

    def _calibrate_interactive(self):
        """Perform interactive calibration with guided head movements"""
        # Define calibration stages with different head positions
        stages = [
            CalibrationStage("prepare", "Please get ready to start calibration...", 2.0),
            CalibrationStage("center", "Look directly at the center of the screen", 5.0, "center"),
            CalibrationStage("left", "Slowly look to your left, then back to center", 3.0, "left"),
            CalibrationStage("right", "Slowly look to your right, then back to center", 3.0, "right"),
            CalibrationStage("up", "Slowly look up, then back to center", 3.0, "up"),
            CalibrationStage("down", "Slowly look down, then back to center", 3.0, "down"),
            CalibrationStage("blink", "Blink normally a few times", 3.0, "blink"),
            CalibrationStage("final", "Look directly at the center again", 3.0, "center"),
        ]
        
        # Calculate total calibration time
        total_time = sum(stage.duration for stage in stages)
        elapsed_time = 0
        
        # Process each stage
        all_data = []
        blink_data = []
        
        try:
            # IMPORTANT: Create window in the main thread, NOT in a worker thread
            # This prevents the trace trap error
            window_name = "Calibration"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, 200, 200)  # Position window in a visible area
            
            # Make sure we can read from the camera
            success, test_img = self.cap.read()
            if not success:
                self.update_status("Cannot read from camera")
                raise Exception("Cannot read from camera")
            
            for i, stage in enumerate(stages):
                stage_start = time.time()
                stage_end = stage_start + stage.duration
                
                self.update_status(stage.instruction)
                
                # Process frames for this stage
                while time.time() < stage_end:
                    # Calculate progress for this stage
                    stage_progress = min(100, (time.time() - stage_start) / stage.duration * 100)
                    # Calculate overall progress - cap at 98% to avoid UI issues
                    overall_progress = min(98, (elapsed_time + (time.time() - stage_start)) / total_time * 100)
                    
                    # Update progress bar
                    self.update_progress(overall_progress)
                    
                    # CRITICAL: Check if window still exists
                    try:
                        # Get window property to check if window exists
                        _ = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    except:
                        # Window was closed, exit calibration
                        raise Exception("Calibration window closed")
                    
                    # Capture and process frame
                    success, img = self.cap.read()
                    if not success:
                        time.sleep(0.01)
                        continue
                    
                    # Use a smaller resolution for processing to improve performance
                    img = cv2.resize(img, (self.process_width, self.process_height))
                    
                    try:
                        img, faces = self.detector.findFaceMesh(img, draw=False)
                    except Exception as e:
                        self.update_status(f"Error in face detection: {str(e)}")
                        faces = []
                    
                    # Display calibration UI feedback
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    progress_text = f"Calibration: {int(overall_progress)}%"
                    cv2.putText(img, stage.instruction, (10, 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, progress_text, (10, 60), font, 0.7, (0, 255, 0), 2)
                    
                    # Draw a progress bar
                    bar_length = 200
                    filled_length = int(bar_length * overall_progress / 100)
                    cv2.rectangle(img, (10, 70), (10 + bar_length, 85), (0, 0, 255), 2)
                    cv2.rectangle(img, (10, 70), (10 + filled_length, 85), (0, 255, 0), -1)
                    
                    # Show current EAR values for visual feedback during calibration
                    if faces:
                        face = faces[0]
                        try:
                            left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
                            right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
                            
                            # Calculate EAR
                            left_ear = self._calculate_ear_enhanced(left_eye)
                            right_ear = self._calculate_ear_enhanced(right_eye)
                            
                            # Display current EAR values
                            ear_text = f"EAR: L={left_ear:.3f}, R={right_ear:.3f}"
                            cv2.putText(img, ear_text, (10, 110), font, 0.6, (255, 255, 0), 2)
                            
                            # Draw eye landmarks for visual feedback - with safety checks
                            try:
                                for point in left_eye:
                                    x, y = int(point[0]), int(point[1])
                                    # Make sure point is within image bounds
                                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                                        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                                for point in right_eye:
                                    x, y = int(point[0]), int(point[1])
                                    # Make sure point is within image bounds
                                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                                        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                            except Exception as e:
                                # If drawing fails, just skip it
                                pass
                            
                            # Calculate head pose
                            try:
                                pitch, yaw, _ = self._calculate_head_pose(face)
                            except Exception as e:
                                # If head pose calculation fails, just skip it
                                pitch, yaw = None, None
                                pass
                            
                            if left_ear > 0.05 and right_ear > 0.05:
                                # For blink calibration stage, collect all data
                                if stage.position == "blink":
                                    blink_data.append({
                                        'left_ear': left_ear,
                                        'right_ear': right_ear,
                                        'time': time.time() - stage_start
                                    })
                                # Only collect data for non-prepare stages
                                elif i > 0:
                                    stage.data.append({
                                        'left_ear': left_ear,
                                        'right_ear': right_ear,
                                        'pitch': pitch if pitch is not None else 0,
                                        'yaw': yaw if yaw is not None else 0,
                                        'position': stage.position,
                                        'time': time.time() - stage_start
                                    })
                        except Exception as e:
                            # Log the specific error for better debugging
                            print(f"Error processing facial landmarks: {str(e)}")
                            pass  # Silent catch to keep calibration running
                    
                    # Show the frame - with error handling
                    try:
                        cv2.imshow(window_name, img)
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC to abort calibration
                            raise KeyboardInterrupt("Calibration aborted by user")
                    except cv2.error as e:
                        # Handle OpenCV errors gracefully
                        self.update_status(f"Error displaying calibration window: {str(e)}")
                        time.sleep(0.01)
                    except Exception as e:
                        # Handle other errors
                        self.update_status(f"Error in calibration: {str(e)}")
                        time.sleep(0.01)
                    
                    # Small delay
                    time.sleep(0.01)
                
                # Update elapsed time
                elapsed_time += stage.duration
                
                # Collect data from this stage
                if stage.data:
                    all_data.extend(stage.data)
            
            # Show 100% completion - keep this simple to avoid hanging
            self.update_progress(100)
            self.update_status("Calibration complete!")
            
            # Final frame display with 100% - use a short timeout for safety
            try:
                success, img = self.cap.read()
                if success:
                    img = cv2.resize(img, (self.process_width, self.process_height))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, "Calibration complete!", (10, 30), font, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, "Calibration: 100%", (10, 60), font, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(img, (10, 70), (10 + 200, 85), (0, 0, 255), 2)
                    cv2.rectangle(img, (10, 70), (10 + 200, 85), (0, 255, 0), -1)
                    cv2.imshow(window_name, img)
                    cv2.waitKey(1)  # Just a quick refresh, don't block too long
                    
                    # Use a safer approach to delay instead of long waitKey
                    delay_end = time.time() + 0.3  # 300ms delay
                    while time.time() < delay_end:
                        cv2.waitKey(1)  # Process events but don't block
                        time.sleep(0.01)
            except Exception as e:
                print(f"Error in final frame display: {str(e)}")
        except Exception as e:
            self.update_status(f"Calibration error: {str(e)}")
            # If there's very little data, return default threshold
            if len(all_data) < 10:
                return 0.22  # default threshold
        finally:
            # CRITICAL: Always ensure proper window cleanup
            # This is essential to prevent the trace trap
            try:
                # Destroy window multiple times to ensure it's gone
                for i in range(5):
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)  # Process window events
                    time.sleep(0.05)
                
                # Also destroy all windows as a last resort
                cv2.destroyAllWindows()
                for i in range(5):
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Error closing calibration windows: {str(e)}")
        
        # Process collected data
        if not all_data:
            self.update_status("Not enough calibration data collected. Using default values.")
            return 0.22  # fallback
        
        try:
            # Extract center position data for threshold calculation
            center_data = [d for d in all_data if d['position'] == 'center']
            
            if not center_data:
                center_data = all_data  # Fallback to all data
            
            # Calculate average EAR for center position
            left_ears = [d['left_ear'] for d in center_data]
            right_ears = [d['right_ear'] for d in center_data]
            
            if not left_ears or not right_ears:
                self.update_status("Insufficient eye data collected. Using default values.")
                return 0.22  # fallback
            
            # Calculate threshold with outlier removal
            left_ears = np.array(left_ears)
            right_ears = np.array(right_ears)
            
            # Remove outliers (values outside 1.5 IQR)
            def remove_outliers(data):
                if len(data) < 2:  # Need at least 2 points for percentile
                    return data
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                return data[(data >= lower_bound) & (data <= upper_bound)]
            
            if len(left_ears) > 4:  # Only apply if we have enough data points
                left_ears = remove_outliers(left_ears)
            if len(right_ears) > 4:
                right_ears = remove_outliers(right_ears)
            
            if len(left_ears) == 0 or len(right_ears) == 0:
                self.update_status("Outlier removal eliminated all data points. Using default values.")
                return 0.22  # fallback
            
            # Calculate threshold - improved method
            avg_ear_when_looking = (np.mean(left_ears) + np.mean(right_ears)) / 2
            
            # If we have blink data, use it to improve threshold calculation
            if blink_data:
                blink_left_ears = np.array([d['left_ear'] for d in blink_data])
                blink_right_ears = np.array([d['right_ear'] for d in blink_data])
                
                # Find the minimum EAR values during blink phase (likely when eyes are more closed)
                min_left = np.percentile(blink_left_ears, 10) if len(blink_left_ears) > 5 else min(blink_left_ears)
                min_right = np.percentile(blink_right_ears, 10) if len(blink_right_ears) > 5 else min(blink_right_ears)
                
                # Average of minimum values
                avg_ear_when_blinking = (min_left + min_right) / 2
                
                # Set threshold between the two values - much closer to the minimum than before
                # This ensures we detect attention more generously
                threshold = avg_ear_when_blinking + (avg_ear_when_looking - avg_ear_when_blinking) * 0.25
                
                # Make sure the threshold isn't too high - cap at 60% of the looking value
                max_threshold = avg_ear_when_looking * 0.6
                threshold = min(threshold, max_threshold)
                
                self.update_status(f"Optimized threshold using blink data: {threshold:.3f}")
            else:
                # Fallback to traditional method - more conservative 
                threshold = avg_ear_when_looking * 0.6  # 60% of average EAR (much more permissive)
            
            # Ensure threshold is reasonable - lowered the upper bound
            if threshold < 0.1:
                self.update_status("Threshold too low. Using adjusted threshold.")
                threshold = 0.1
            elif threshold > 0.3:  # Reduced from 0.5 to 0.3 as upper limit
                self.update_status("Threshold too high. Using adjusted threshold.")
                threshold = 0.3
            
            # Also update head pose thresholds based on calibration data
            try:
                pitch_values = [abs(d['pitch']) for d in all_data if 'pitch' in d and d['pitch'] is not None]
                yaw_values = [abs(d['yaw']) for d in all_data if 'yaw' in d and d['yaw'] is not None]
                
                if pitch_values and yaw_values:
                    # Set thresholds at 90th percentile of observed values (more permissive)
                    pitch_threshold = np.percentile(pitch_values, 90)
                    yaw_threshold = np.percentile(yaw_values, 90)
                    
                    # Update the head pose thresholds - make it more generous
                    self.head_pose_threshold = max(35, min(60, (pitch_threshold + yaw_threshold) / 2))
                    self.update_status(f"Updated head pose threshold: {self.head_pose_threshold:.1f} degrees")
            except Exception as e:
                self.update_status(f"Could not update head pose thresholds: {str(e)}")
            
            # Set adaptive thresholding for continuous adjustments
            self.adaptive_threshold_enabled = True
            
            # Initialize the min/max history with calibration values for faster adaptation
            self.ear_min_history.clear()
            self.ear_max_history.clear()
            
            for _ in range(5):
                self.ear_min_history.append(avg_ear_when_blinking if 'avg_ear_when_blinking' in locals() else threshold * 0.8)
                self.ear_max_history.append(avg_ear_when_looking)
            
            return threshold
            
        except Exception as e:
            self.update_status(f"Error processing calibration data: {str(e)}")
            return 0.22  # fallback threshold

class CalibrationWorker(QObject):
    finished = pyqtSignal(float)  # Will emit the calculated threshold
    progress_update = pyqtSignal(float)  # Will emit progress percentage
    status_update = pyqtSignal(str)  # Will emit status messages

    def __init__(self, eye_tracker):
        super().__init__()
        self.eye_tracker = eye_tracker

    def run(self):
        # Store original callbacks
        original_status_callback = self.eye_tracker.status_callback
        
        # Set our callbacks to forward signals
        self.eye_tracker.set_status_callback(lambda msg: self.status_update.emit(msg))
        self.eye_tracker.set_progress_callback(lambda progress: self.progress_update.emit(progress))
        
        try:
            threshold = self.eye_tracker._calibrate_interactive()
            self.finished.emit(threshold)
        finally:
            # Restore original callback
            self.eye_tracker.set_status_callback(original_status_callback)

if __name__ == "__main__":
    # Create QApplication instance for PyQt to work properly
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Check if running on Raspberry Pi
    import platform
    is_raspberry_pi = platform.machine().startswith('arm')
    
    # Enable lightweight mode by default on Raspberry Pi
    eye_tracker = EyeTracker(lightweight_mode=is_raspberry_pi)
    try:
        print(f"Starting eye tracking test in {'lightweight' if eye_tracker.lightweight_mode else 'full'} mode...")
        print("Press Ctrl+C to calibrate, then press Ctrl+C again to exit")
        
        # Track previous state to only print when state changes
        prev_looking = False
        prev_present = False
        
        while True:
            user_present = eye_tracker.is_user_present()
            looking_at_camera = False
            
            if user_present:
                looking_at_camera = eye_tracker.is_looking_at_camera()
                
                # Only print when state changes or every 30 iterations
                if looking_at_camera != prev_looking or user_present != prev_present:
                    if looking_at_camera:
                        print(f"\033[92m👁️  LOOKING AT CAMERA (FPS: {eye_tracker.get_fps()})\033[0m")  # Green text
                    else:
                        print(f"\033[91m❌  NOT LOOKING AT CAMERA (FPS: {eye_tracker.get_fps()})\033[0m")  # Red text
                
                prev_looking = looking_at_camera
            else:
                if user_present != prev_present:
                    print(f"\033[93m⚠️  NO USER DETECTED (FPS: {eye_tracker.get_fps()})\033[0m")  # Yellow text
            
            prev_present = user_present
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStarting calibration...")
        # Use direct calibration instead of Qt-based for command-line usage
        eye_tracker.calibrate()
        try:
            print("Press Ctrl+C to exit...")
            
            # Reset state tracking after calibration
            prev_looking = False
            prev_present = False
            
            while True:
                user_present = eye_tracker.is_user_present()
                looking_at_camera = False
                
                if user_present:
                    looking_at_camera = eye_tracker.is_looking_at_camera()
                    
                    # Only print when state changes or every 30 iterations
                    if looking_at_camera != prev_looking or user_present != prev_present:
                        if looking_at_camera:
                            print(f"\033[92m👁️  LOOKING AT CAMERA (FPS: {eye_tracker.get_fps()})\033[0m")  # Green text
                        else:
                            print(f"\033[91m❌  NOT LOOKING AT CAMERA (FPS: {eye_tracker.get_fps()})\033[0m")  # Red text
                    
                    prev_looking = looking_at_camera
                else:
                    if user_present != prev_present:
                        print(f"\033[93m⚠️  NO USER DETECTED (FPS: {eye_tracker.get_fps()})\033[0m")  # Yellow text
                
                prev_present = user_present
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping eye tracking...")
    finally:
        eye_tracker.release()

