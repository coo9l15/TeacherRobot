import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import time
from collections import deque
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication
import sys

class CalibrationStage:
    def __init__(self, name, instruction, duration=3.0, position="center"):
        self.name = name
        self.instruction = instruction
        self.duration = duration
        self.position = position  # center, left, right, up, down
        self.data = []

class EyeTracker:
    def __init__(self, smoothing_window=5):  # Increased smoothing window
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.detector = FaceMeshDetector(
            maxFaces=1,
            minDetectionCon=0.6,  # Increased detection confidence
            minTrackCon=0.6       # Increased tracking confidence
        )
        
        self.smoothing_window = smoothing_window
        self.ear_history = deque(maxlen=smoothing_window)
        
        # Default thresholds, will be calibrated
        self.ear_threshold = 0.22 # Default threshold
        self.head_pose_threshold = 35
        
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
                
            img = cv2.resize(img, (320, 240))
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
                    
                    face = faces[0]
                    try:
                        left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
                        right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
                    except KeyError:
                        self.user_present = False # Consider no user if landmarks are missing
                        self.latest_result = False
                        # Continue to update FPS
                    else:
                        left_ear = self._calculate_ear(left_eye)
                        right_ear = self._calculate_ear(right_eye)
                        
                        pitch, yaw, _ = self._calculate_head_pose(face)
                        
                        if pitch is not None and yaw is not None:
                            if abs(pitch) > self.head_pose_threshold or abs(yaw) > self.head_pose_threshold:
                                self.latest_result = False
                            else:
                                avg_ear = (left_ear + right_ear) / 2
                                self.ear_history.append(avg_ear)
                                smoothed_ear = np.mean(self.ear_history) if self.ear_history else avg_ear
                                self.latest_result = smoothed_ear > self.ear_threshold
                        else: # Head pose couldn't be calculated
                            self.latest_result = False
            
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            
            time.sleep(0.001)

    def is_looking_at_camera(self):
        """Check if the user is looking at the camera"""
        with self.lock:
            return self.latest_result if self.user_present else False

    def is_user_present(self):
        """Check if a user is present in front of the camera"""
        with self.lock:
            return self.user_present

    def _calculate_head_pose(self, face):
        """Optimized head pose calculation"""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye
                (225.0, 170.0, -135.0),   # Right eye
            ])
            image_points = np.array([
                face[1], face[152], face[33], face[263],
            ], dtype="double")
            camera_matrix = np.array(
                [[640, 0, 160],[0, 640, 120],[0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4,1))
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            return euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]
        except Exception:
            return None, None, None

    def _calculate_ear(self, eye):
        """Optimized EAR calculation"""
        try:
            eye = np.array(eye)
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            ear = (A + B) / (2.0 * C)
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

    def _calibrate_interactive(self):
        """Perform interactive calibration with guided head movements"""
        # Define calibration stages with different head positions
        stages = [
            CalibrationStage("prepare", "Please get ready to start calibration...", 2.0),
            CalibrationStage("center", "Look directly at the center of the screen", 4.0, "center"),
            CalibrationStage("left", "Slowly look to your left, then back to center", 4.0, "left"),
            CalibrationStage("right", "Slowly look to your right, then back to center", 4.0, "right"),
            CalibrationStage("up", "Slowly look up, then back to center", 4.0, "up"),
            CalibrationStage("down", "Slowly look down, then back to center", 4.0, "down"),
            CalibrationStage("final", "Look directly at the center again", 3.0, "center"),
        ]
        
        # Calculate total calibration time
        total_time = sum(stage.duration for stage in stages)
        elapsed_time = 0
        
        # Process each stage
        all_data = []
        for i, stage in enumerate(stages):
            stage_start = time.time()
            stage_end = stage_start + stage.duration
            
            self.update_status(stage.instruction)
            
            # Process frames for this stage
            while time.time() < stage_end:
                # Calculate progress for this stage
                stage_progress = min(100, (time.time() - stage_start) / stage.duration * 100)
                # Calculate overall progress
                overall_progress = min(100, (elapsed_time + (time.time() - stage_start)) / total_time * 100)
                
                # Update progress bar
                self.update_progress(overall_progress)
                
                # Capture and process frame
                success, img = self.cap.read()
                if not success:
                    time.sleep(0.01)
                    continue
                
                img = cv2.resize(img, (320, 240))
                img, faces = self.detector.findFaceMesh(img, draw=False)
                
                if faces:
                    face = faces[0]
                    try:
                        left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
                        right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
                        
                        # Calculate EAR
                        left_ear = self._calculate_ear(left_eye)
                        right_ear = self._calculate_ear(right_eye)
                        
                        # Calculate head pose
                        pitch, yaw, _ = self._calculate_head_pose(face)
                        
                        if left_ear > 0.05 and right_ear > 0.05:
                            # Only collect data for non-prepare stages
                            if i > 0:
                                stage.data.append({
                                    'left_ear': left_ear,
                                    'right_ear': right_ear,
                                    'pitch': pitch,
                                    'yaw': yaw,
                                    'position': stage.position,
                                    'time': time.time() - stage_start
                                })
                    except Exception as e:
                        pass
                
                # Small delay
                time.sleep(0.01)
            
            # Update elapsed time
            elapsed_time += stage.duration
            
            # Collect data from this stage
            if stage.data:
                all_data.extend(stage.data)
        
        # Process collected data
        if not all_data:
            self.update_status("Calibration failed! Using default values.")
            return 0.22  # fallback
        
        # Extract center position data for threshold calculation
        center_data = [d for d in all_data if d['position'] == 'center']
        
        if not center_data:
            center_data = all_data  # Fallback to all data
        
        # Calculate average EAR for center position
        left_ears = [d['left_ear'] for d in center_data]
        right_ears = [d['right_ear'] for d in center_data]
        
        # Calculate threshold
        avg_ear = (np.mean(left_ears) + np.mean(right_ears)) / 2
        threshold = avg_ear * 0.85  # 85% of average EAR when looking at center
        
        # Update progress to 100%
        self.update_progress(100)
        
        return threshold

    def calibrate_for_student(self, student_id, student_name=""):
        """Calibrate for a specific student and store their profile"""
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
        
        self.is_running = True
        self.start_processing()
        return threshold

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
    
    eye_tracker = EyeTracker()
    try:
        print("Starting eye tracking test...")
        print("Press Ctrl+C to calibrate, then press Ctrl+C again to exit")
        while True:
            if eye_tracker.is_user_present():
                if eye_tracker.is_looking_at_camera():
                    print(f"User is looking at camera (FPS: {eye_tracker.get_fps()})")
                else:
                    print(f"User is not looking at camera (FPS: {eye_tracker.get_fps()})")
            else:
                print(f"No user detected (FPS: {eye_tracker.get_fps()})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStarting calibration...")
        # Use direct calibration instead of Qt-based for command-line usage
        eye_tracker.calibrate()
        try:
            print("Press Ctrl+C to exit...")
            while True:
                if eye_tracker.is_user_present():
                    if eye_tracker.is_looking_at_camera():
                        print(f"User is looking at camera (FPS: {eye_tracker.get_fps()})")
                    else:
                        print(f"User is not looking at camera (FPS: {eye_tracker.get_fps()})")
                else:
                    print(f"No user detected (FPS: {eye_tracker.get_fps()})")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping eye tracking...")
    finally:
        eye_tracker.release()

