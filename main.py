import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


class MemeMatcher:
    def __init__(self, assets_folder="assets"):
        # Initialize MediaPipe Face Landmarker
        self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=self._download_face_model()
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        # Initialize MediaPipe Hand Landmarker
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=self._download_hand_model()
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
        )

        # MediaPipe Face Mesh landmark indices for key facial features
        # Eyes
        self.LEFT_EYE_UPPER = [159, 145, 158]
        self.LEFT_EYE_LOWER = [23, 27, 133]
        self.RIGHT_EYE_UPPER = [386, 374, 385]
        self.RIGHT_EYE_LOWER = [253, 257, 362]

        # Eyebrows
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336]

        # Mouth outer
        self.MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
        # Mouth inner
        self.MOUTH_INNER = [78, 308, 95, 88]

        # Nose tip
        self.NOSE_TIP = 4

        # Frame counter for video processing
        self.frame_counter = 0

        # Load memes
        self.memes = []
        self.meme_features = []
        self.load_memes(assets_folder)

    def _download_face_model(self):
        """Download and return path to face landmarker model"""
        import os
        import subprocess

        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print("Face model downloaded successfully!")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download model. Please download manually from {url}")
        return model_path

    def _download_hand_model(self):
        """Download and return path to hand landmarker model"""
        import os
        import subprocess

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                subprocess.run(['curl', '-L', url, '-o', model_path], check=True, capture_output=True)
                print("Hand model downloaded successfully!")
            except subprocess.CalledProcessError:
                raise RuntimeError(f"Failed to download hand model. Please download manually from {url}")
        return model_path

    def load_memes(self, folder):
        """Load all meme images from assets folder"""
        assets_path = Path(folder)
        # Create face landmarker for static images with lower thresholds
        static_face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=self._download_face_model()
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3
            )
        )

        # Create hand landmarker for static images
        static_hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=self._download_hand_model()
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3
            )
        )

        # Load both JPG and PNG files
        image_files = list(assets_path.glob("*.jpg")) + list(assets_path.glob("*.png")) + list(
            assets_path.glob("*.jpeg"))
        for img_file in sorted(image_files):
            img = cv2.imread(str(img_file))
            if img is not None:
                # Extract facial and hand features from meme
                features = self.extract_face_features(img, static_face_landmarker, static_hand_landmarker,
                                                      is_static=True)
                if features:
                    self.memes.append({
                        'image': img,
                        'name': img_file.stem.replace('_', ' ').title(),
                        'path': str(img_file)
                    })
                    self.meme_features.append(features)
                    print(f"Loaded: {img_file.name}")
                else:
                    print(f"Skipping {img_file.name}: No face detected")

        static_face_landmarker.close()
        static_hand_landmarker.close()
        print(f"\nTotal memes loaded: {len(self.memes)}")

    def extract_face_features(self, image, face_landmarker_instance=None, hand_landmarker_instance=None,
                              is_static=False):
        """Extract facial and hand features using MediaPipe"""
        if face_landmarker_instance is None:
            face_landmarker_instance = self.face_mesh
        if hand_landmarker_instance is None:
            hand_landmarker_instance = self.hand_detector

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Process face based on mode
        if is_static:
            face_results = face_landmarker_instance.detect(mp_image)
        else:
            self.frame_counter += 1
            face_results = face_landmarker_instance.detect_for_video(mp_image, self.frame_counter)

        if not face_results.face_landmarks:
            return None

        # Detect hands
        if is_static:
            hand_results = hand_landmarker_instance.detect(mp_image)
        else:
            hand_results = hand_landmarker_instance.detect_for_video(mp_image, self.frame_counter)

        # Count hands detected and check hand position
        num_hands = len(hand_results.hand_landmarks) if hand_results.hand_landmarks else 0

        # Check if hand is raised (at or above face level for "cheers" gesture)
        hand_raised = False
        if num_hands > 0 and hand_results.hand_landmarks:
            landmarks_face = face_results.face_landmarks[0]
            face_center_y = np.mean([l.y for l in landmarks_face])
            face_top_y = min([l.y for l in landmarks_face])

            for hand_landmarks in hand_results.hand_landmarks:
                wrist_y = hand_landmarks[0].y
                middle_finger_y = hand_landmarks[12].y

                if middle_finger_y < face_center_y + 0.2 or wrist_y < face_top_y + 0.3:
                    hand_raised = True
                    break

        landmarks = face_results.face_landmarks[0]
        h, w, _ = image.shape

        def get_point(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y])

        # Calculate eye aspect ratios (EAR) - measures eye openness
        def calc_eye_aspect_ratio(upper_indices, lower_indices):
            upper_pts = [get_point(i) for i in upper_indices]
            lower_pts = [get_point(i) for i in lower_indices]

            # Vertical distances
            vert_dist = sum([np.linalg.norm(upper_pts[i] - lower_pts[i]) for i in range(len(upper_pts))]) / len(
                upper_pts)

            # Horizontal distance (eye width)
            horiz_dist = np.linalg.norm(get_point(upper_indices[0]) - get_point(upper_indices[-1]))

            return vert_dist / (horiz_dist + 1e-6)

        left_ear = calc_eye_aspect_ratio(self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER)
        right_ear = calc_eye_aspect_ratio(self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER)
        avg_ear = (left_ear + right_ear) / 2.0

        # Calculate mouth aspect ratio
        mouth_outer_pts = [get_point(i) for i in self.MOUTH_OUTER]
        mouth_top = get_point(13)  # Upper lip
        mouth_bottom = get_point(14)  # Lower lip

        # Mouth vertical opening
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)

        # Mouth horizontal width
        mouth_left = get_point(61)
        mouth_right = get_point(291)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)

        mouth_ar = mouth_height / (mouth_width + 1e-6)

        # Calculate mouth width ratio
        inner_mouth_left = get_point(78)
        inner_mouth_right = get_point(308)
        inner_mouth_width = np.linalg.norm(inner_mouth_left - inner_mouth_right)
        mouth_width_ratio = inner_mouth_width / (mouth_width + 1e-6)

        # Eyebrow positions
        left_brow_pts = [get_point(i) for i in self.LEFT_EYEBROW]
        right_brow_pts = [get_point(i) for i in self.RIGHT_EYEBROW]

        left_brow_y = np.mean([p[1] for p in left_brow_pts])
        right_brow_y = np.mean([p[1] for p in right_brow_pts])

        # Eye centers for reference
        left_eye_center_y = np.mean([get_point(i)[1] for i in self.LEFT_EYE_UPPER + self.LEFT_EYE_LOWER])
        right_eye_center_y = np.mean([get_point(i)[1] for i in self.RIGHT_EYE_UPPER + self.RIGHT_EYE_LOWER])

        # Eyebrow height relative to eyes
        left_brow_height = left_eye_center_y - left_brow_y
        right_brow_height = right_eye_center_y - right_brow_y
        avg_brow_height = (left_brow_height + right_brow_height) / 2.0

        # Mouth corners (smile/frown detection)
        mouth_left_corner = get_point(61)
        mouth_right_corner = get_point(291)
        mouth_center_y = (mouth_left_corner[1] + mouth_right_corner[1]) / 2.0

        # Nose tip position
        nose_tip = get_point(self.NOSE_TIP)

        # Mouth corner elevation (relative to nose)
        mouth_elevation = nose_tip[1] - mouth_center_y

        # Facial expression features
        features = {
            # Eye features
            'eye_openness': avg_ear,
            'left_eye_open': left_ear,
            'right_eye_open': right_ear,
            'eyes_symmetry': abs(left_ear - right_ear),

            # Mouth features
            'mouth_openness': mouth_ar,
            'mouth_width': mouth_width,
            'mouth_width_ratio': mouth_width_ratio,
            'mouth_elevation': mouth_elevation,

            # Eyebrow features
            'eyebrow_height': avg_brow_height,
            'left_brow_height': left_brow_height,
            'right_brow_height': right_brow_height,
            'brow_symmetry': abs(left_brow_height - right_brow_height),

            # Hand features (for Leo's cheers gesture)
            'num_hands': num_hands,
            'hand_raised': 1.0 if hand_raised else 0.0,

            # Overall expression indicators
            'surprise_score': avg_ear * avg_brow_height * mouth_ar,  # Wide eyes + raised brows + open mouth
            'smile_score': mouth_width_ratio * (1.0 - mouth_ar),  # Wide mouth + closed mouth
            'concern_score': avg_brow_height * (1.0 - mouth_elevation),  # Raised brows + downturned mouth
            'cheers_score': mouth_width_ratio * (1.0 - mouth_ar) * (1.0 if hand_raised else 0.0),  # Smile + raised hand
        }

        return features

    def compute_similarity(self, features1, features2):
        """Compute similarity score between two feature sets using improved matching"""
        if features1 is None or features2 is None:
            return 0.0

        score = 0.0

        # READJUST THE WEIGHTSS IF THISSS IF IT DOESNT WORK WELL
        feature_weights = {
            # Expression-specific features (highest weight)
            'surprise_score': 25.0,
            'smile_score': 20.0,
            'concern_score': 20.0,
            'cheers_score': 30.0,

            # Hand features
            'hand_raised': 25.0,
            'num_hands': 15.0,

            # Eye features
            'eye_openness': 20.0,
            'eyes_symmetry': 10.0,

            # Mouth features
            'mouth_openness': 25.0,
            'mouth_width_ratio': 20.0,
            'mouth_elevation': 15.0,

            # Eyebrow features
            'eyebrow_height': 20.0,
            'brow_symmetry': 10.0,
        }

        # Calculate similarity for each feature
        for feature, weight in feature_weights.items():
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]

                # Normalize the difference
                diff = abs(val1 - val2)

                # Use exponential decay for similarity (TODO: CHECK THIS AGAAIIIN )
                if feature in ['surprise_score', 'smile_score', 'concern_score', 'cheers_score']:
                    # Expression scores need special handling
                    similarity = np.exp(-diff * 10)  # More sensitive
                elif feature in ['hand_raised', 'num_hands']:
                    # Hand features are binary/discrete, so use stricter matching
                    similarity = np.exp(-diff * 15)  # Very sensitive
                else:
                    similarity = np.exp(-diff * 5)

                score += weight * similarity

        return score

    def find_best_match(self, user_features):
        """Find the meme that best matches the user's face"""
        if user_features is None:
            return None, 0.0

        best_idx = -1
        best_score = -1.0

        for idx, meme_features in enumerate(self.meme_features):
            score = self.compute_similarity(user_features, meme_features)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx != -1:
            return self.memes[best_idx], best_score
        return None, 0.0

    def run(self):
        """Main loop - capture webcam and match memes"""
        cap = cv2.VideoCapture(0)

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\n🎥 Camera started! Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Extract features from user's face
            user_features = self.extract_face_features(frame)

            # Find best matching meme
            best_meme, score = self.find_best_match(user_features)

            # Create display frame
            h, w = frame.shape[:2]

            if best_meme is not None:
                # Resize meme to match camera frame height
                meme_img = best_meme['image'].copy()
                meme_h, meme_w = meme_img.shape[:2]
                scale = h / meme_h
                new_w = int(meme_w * scale)
                meme_resized = cv2.resize(meme_img, (new_w, h))

                display = np.zeros((h, w + new_w, 3), dtype=np.uint8)
                display[:, :w] = frame
                display[:, w:w + new_w] = meme_resized

                #  text overlay with background
                cv2.rectangle(display, (5, 5), (200, 45), (0, 0, 0), -1)
                cv2.putText(display, "YOU", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                cv2.rectangle(display, (w + 5, 5), (w + new_w - 5, 75), (0, 0, 0), -1)
                cv2.putText(display, best_meme['name'], (w + 10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Match: {score:.1f}", (w + 10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                display = frame
                cv2.putText(display, "No face detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Meme Matcher - Press Q to quit', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print(" Meme Matcher Starting...\n")
    matcher = MemeMatcher(assets_folder="assets")
    matcher.run()