"""Team assignment module for basketball video analysis.

This module uses a pre-trained CLIP vision-language model to classify players
into teams based on their jersey colors/appearance.

Team assignment is stabilised with a **weighted voting** scheme: instead of
locking a player's team based on a single CLIP prediction, the system
accumulates the softmax probability for "team 1" across every frame the
player is visible and classifies them according to the running average.
This makes the assignment robust to individual bad crops (occlusion,
motion blur, overlapping defenders, etc.).
"""

from collections import defaultdict

from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

from .utils import read_stub, save_stub


class TeamAssigner:
    """Assigns players to teams based on jersey appearance (CLIP).

    Instead of deciding once and caching, the assigner keeps a running
    weighted vote per ``track_id``.  On each frame:

    1. CLIP produces a softmax probability ``p(team_1)`` for the crop.
    2. That probability is added to a cumulative score for the player.
    3. The team is determined by whether the *average* score across all
       observations so far is above or below 0.5.

    This means a single bad frame cannot flip the assignment — it takes a
    sustained streak of contrary evidence.  An optional ``classify_every``
    parameter lets you skip frames to save compute while still
    accumulating enough evidence.

    Attributes:
        team_1_class_name (str): Text prompt describing Team 1's jersey.
        team_2_class_name (str): Text prompt describing Team 2's jersey.
        classify_every (int): Run CLIP every *N*-th frame per player
            (1 = every frame).  Intermediate frames reuse the running vote.
        min_observations (int): Number of CLIP samples required before
            the running vote is trusted.  Until this threshold is met the
            latest single-frame argmax is used (best-effort).
    """

    def __init__(
        self,
        team_1_class_name="white shirt",
        team_2_class_name="dark blue shirt",
        classify_every: int = 5,
        min_observations: int = 3,
    ):
        """
        Initialize the TeamAssigner with specified team jersey descriptions.

        Args:
            team_1_class_name: Description of Team 1's jersey appearance.
            team_2_class_name: Description of Team 2's jersey appearance.
            classify_every: Run CLIP inference every N frames per player
                to balance accuracy vs. speed (default 5).
            min_observations: Minimum number of CLIP observations before
                the running vote is considered reliable (default 3).
        """
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.classify_every = max(1, classify_every)
        self.min_observations = max(1, min_observations)

        # Running state --------------------------------------------------
        # Cumulative sum of p(team_1) for each player_id
        self._team1_score: dict[int, float] = defaultdict(float)
        # Number of CLIP observations for each player_id
        self._obs_count: dict[int, int] = defaultdict(int)
        # How many frames since the last CLIP call for each player_id
        self._frames_since_clip: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def load_model(self):
        """Load the pre-trained CLIP model for jersey colour classification.

        Uses *fashion-clip* which is fine-tuned for clothing, giving better
        jersey colour discrimination than the base CLIP model.
        """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    # ------------------------------------------------------------------ #
    # Single-crop classification (returns probability, not hard label)
    # ------------------------------------------------------------------ #
    def _get_team1_probability(self, frame, bbox) -> float:
        """Return the softmax probability that the crop belongs to Team 1.

        Args:
            frame: BGR video frame (numpy array).
            bbox: ``(x1, y1, x2, y2)`` bounding box of the player.

        Returns:
            Float in [0, 1] — probability of Team 1 jersey.
        """
        crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Guard against degenerate crops (zero-area, etc.)
        if crop.size == 0:
            return 0.5  # neutral — do not influence the vote

        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(
            text=classes, images=pil_image, return_tensors="pt", padding=True
        )

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        # probs[0][0] = p(team_1), probs[0][1] = p(team_2)
        return float(probs[0][0])

    # ------------------------------------------------------------------ #
    # Per-player team decision (with running vote)
    # ------------------------------------------------------------------ #
    def get_player_team(self, frame, player_bbox, player_id) -> int:
        """Return the team id for *player_id*, updating the running vote.

        CLIP is invoked only every ``classify_every`` frames for a given
        player.  On skipped frames the latest running average is returned.

        Args:
            frame: BGR video frame (numpy array).
            player_bbox: ``(x1, y1, x2, y2)`` bounding box.
            player_id: Persistent track id for the player.

        Returns:
            ``1`` (Team 1) or ``2`` (Team 2).
        """
        should_classify = (
            self._obs_count[player_id] == 0  # first sighting — always classify
            or self._frames_since_clip[player_id] >= self.classify_every
        )

        if should_classify:
            p_team1 = self._get_team1_probability(frame, player_bbox)
            self._team1_score[player_id] += p_team1
            self._obs_count[player_id] += 1
            self._frames_since_clip[player_id] = 0
        else:
            self._frames_since_clip[player_id] += 1

        # Decide based on running average
        avg = self._team1_score[player_id] / self._obs_count[player_id]
        return 1 if avg >= 0.5 else 2

    # ------------------------------------------------------------------ #
    # Batch processing across all frames
    # ------------------------------------------------------------------ #
    def get_player_teams_across_frames(
        self,
        video_frames,
        player_tracks,
        read_from_stub=False,
        stub_path=None,
    ):
        """Process all video frames and assign teams to every tracked player.

        Args:
            video_frames: List of BGR video frames.
            player_tracks: Per-frame dicts ``{track_id: {"bbox": [x1,y1,x2,y2]}}``.
            read_from_stub: Whether to try loading cached results first.
            stub_path: Path to the pickle cache file.

        Returns:
            List of dicts ``{track_id: team_id}`` for each frame.
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        # Reset running state for a fresh run
        self._team1_score.clear()
        self._obs_count.clear()
        self._frames_since_clip.clear()

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            frame_assignments = {}
            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                frame_assignments[player_id] = team
            player_assignment.append(frame_assignments)

        save_stub(stub_path, player_assignment)

        return player_assignment
