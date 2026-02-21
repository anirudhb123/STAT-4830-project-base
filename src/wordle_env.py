"""
Wordle environment wrapper for Prime Intellect's verifiers.

This module provides an adapter between Prime Intellect's Wordle environment
and our RL training infrastructure.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Optional imports for when dependencies are available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch/numpy not available. Install requirements.txt")


@dataclass
class WordleState:
    """State representation for Wordle game."""
    conversation_history: str
    turn_number: int
    game_complete: bool
    target_word: Optional[str] = None
    previous_guesses: List[str] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)


class WordleEnvironmentWrapper:
    """
    Wrapper for Prime Intellect's Wordle environment.
    
    This adapter makes the Wordle environment compatible with our
    PPO training pipeline by:
    - Converting text states to numeric representations
    - Parsing XML responses
    - Computing rewards from the verifiers rubric
    """
    
    def __init__(
        self,
        num_episodes: int = 100,
        max_turns: int = 6,
        seed: Optional[int] = None
    ):
        """
        Initialize Wordle environment.
        
        Args:
            num_episodes: Number of episodes to generate
            max_turns: Maximum turns per episode (standard Wordle is 6)
            seed: Random seed for reproducibility
        """
        self.num_episodes = num_episodes
        self.max_turns = max_turns
        self.seed = seed
        
        # Try to load Prime Intellect environment
        self.prime_env = None
        self._load_prime_environment()
        
        # State tracking
        self.current_episode = 0
        self.current_state = None
        self.episode_data = None
        
    def _load_prime_environment(self):
        """Load the Prime Intellect Wordle environment."""
        try:
            import os
            from verifiers import load_environment
            
            # Set NLTK data path to venv
            nltk_data_path = os.path.join(os.getcwd(), '.venv', 'nltk_data')
            if os.path.exists(nltk_data_path):
                os.environ['NLTK_DATA'] = nltk_data_path
            
            # Load environment (it takes env_id only, no args)
            self.prime_env = load_environment("wordle")
            print("✓ Successfully loaded Prime Intellect Wordle environment")
            print(f"  Training examples: {len(self.prime_env.dataset) if hasattr(self.prime_env, 'dataset') else 'N/A'}")
            print(f"  Eval examples: {len(self.prime_env.eval_dataset) if hasattr(self.prime_env, 'eval_dataset') else 'N/A'}")
        except ImportError:
            print("WARNING: Prime Intellect verifiers not installed. Using mock environment.")
            print("Install with: uv pip install verifiers>=0.1.9")
            self.prime_env = None
        except Exception as e:
            print(f"WARNING: Could not load Wordle environment: {e}")
            print("Using mock environment for testing.")
            self.prime_env = None
    
    def reset(self) -> WordleState:
        """
        Reset environment to start a new episode.
        
        Returns:
            Initial state for the episode
        """
        if self.prime_env is not None:
            # Prime Intellect uses dataset-based episodes
            # Get a random episode from the dataset
            if hasattr(self.prime_env, 'dataset') and len(self.prime_env.dataset) > 0:
                import random
                episode_idx = random.randint(0, len(self.prime_env.dataset) - 1)
                episode = self.prime_env.dataset[episode_idx]
                
                # Handle prompt (can be string or list)
                initial_prompt = episode.get('prompt', episode.get('input', ''))
                if isinstance(initial_prompt, list):
                    initial_prompt = '\n'.join(str(x) for x in initial_prompt)
                elif not isinstance(initial_prompt, str):
                    initial_prompt = str(initial_prompt)
                
                # Handle target (can be string or list)
                target = episode.get('target', episode.get('answer', ''))
                if isinstance(target, list):
                    target = target[0] if len(target) > 0 else "CRANE"
                target = str(target).upper() if target else "CRANE"
                
                self.current_state = WordleState(
                    conversation_history=initial_prompt,
                    turn_number=0,
                    game_complete=False,
                    target_word=target
                )
                self.current_episode_data = episode
            else:
                # Fallback if no dataset
                self.current_state = WordleState(
                    conversation_history="Guess a 5-letter word. You have 6 attempts.",
                    turn_number=0,
                    game_complete=False,
                    target_word="CRANE"
                )
        else:
            # Mock environment for testing
            import random
            mock_targets = ["CRANE", "SLATE", "TRACE", "AUDIO", "PLANT", "HEART", "LIGHT", "DREAM"]
            self.current_state = WordleState(
                conversation_history="Guess a 5-letter word. You have 6 attempts.",
                turn_number=0,
                game_complete=False,
                target_word=random.choice(mock_targets)
            )
        
        return self.current_state
    
    def step(
        self,
        action: str
    ) -> Tuple[WordleState, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The guess (should be formatted as XML with <think> and <guess>)
            
        Returns:
            next_state: Updated state after the action
            reward: Reward for this step
            done: Whether the episode is complete
            info: Additional information
        """
        # Parse the action to extract the guess
        guess = self._parse_guess(action)
        
        if self.prime_env is not None:
            # Prime Intellect environment - simulate step using TextArena backend
            # Note: Real Prime Intellect workflow uses generate() with LLM client
            # For RL training, we simulate step-by-step interaction
            reward, done, info = self._simulate_prime_step(guess)
            
            # Create new lists by concatenating
            new_guesses = self.current_state.previous_guesses + [guess]
            new_feedback = self.current_state.feedback_history + [info["feedback"]]
            
            self.current_state = WordleState(
                conversation_history=self.current_state.conversation_history + f"\n{info['feedback']}",
                turn_number=self.current_state.turn_number + 1,
                game_complete=done,
                target_word=self.current_state.target_word,
                previous_guesses=new_guesses,
                feedback_history=new_feedback
            )
        else:
            # Mock environment for testing
            reward, done, info = self._mock_step(guess)
            
            # Create new lists by concatenating
            new_guesses = self.current_state.previous_guesses + [guess]
            new_feedback = self.current_state.feedback_history + [info["feedback"]]
            
            self.current_state = WordleState(
                conversation_history=self.current_state.conversation_history + f"\nGuess: {guess}",
                turn_number=self.current_state.turn_number + 1,
                game_complete=done,
                target_word=self.current_state.target_word,
                previous_guesses=new_guesses,
                feedback_history=new_feedback
            )
        
        return self.current_state, reward, done, info
    
    def _simulate_prime_step(self, guess: str) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Simulate a step using Prime Intellect's Wordle game logic.
        
        This replicates the Wordle game mechanics and reward structure
        used by Prime Intellect's environment.
        """
        target = self.current_state.target_word
        turn = self.current_state.turn_number + 1
        
        # Check if guess is correct
        correct = (guess.upper() == target.upper()) if target else False
        
        # Generate Wordle feedback
        feedback = self._generate_wordle_feedback(guess, target)
        
        # Compute rewards matching Prime Intellect's rubric
        correct_answer = 1.0 if correct else 0.0
        
        # Partial credit from feedback (greens and yellows)
        partial_answer = self._compute_partial_credit(feedback)
        
        # Length bonus (solving in fewer turns is better)
        length_bonus = (self.max_turns - turn + 1) / self.max_turns if correct else 0.0
        
        # Format reward (1.0 if valid guess, 0.0 otherwise)
        format_reward = 1.0 if len(guess) == 5 and guess.isalpha() else 0.0
        
        # Total reward
        reward = correct_answer + partial_answer * 0.3 + length_bonus * 0.5 + format_reward * 0.1
        
        # Check if done
        done = correct or turn >= self.max_turns
        
        info = {
            "correct_answer": correct_answer,
            "partial_answer": partial_answer,
            "length_bonus": length_bonus,
            "format_reward": format_reward,
            "guess": guess,
            "turn": turn,
            "feedback": feedback,
            "target": target
        }
        
        return reward, done, info
    
    def _compute_partial_credit(self, feedback: str) -> float:
        """Compute partial credit from Wordle feedback (0.0 to 1.0)."""
        if not feedback:
            return 0.0
        
        # Count greens and yellows
        greens = feedback.count("GREEN")
        yellows = feedback.count("YELLOW")
        
        # Normalize: 5 greens = 1.0, yellows count less
        return (greens * 0.2 + yellows * 0.1)
    
    def _generate_wordle_feedback(self, guess: str, target: str) -> str:
        """Generate Wordle-style feedback matching Prime Intellect format."""
        if not target or len(guess) != len(target):
            return "Invalid guess"
        
        guess = guess.upper()
        target = target.upper()
        
        feedback = []
        target_letters = list(target)
        
        # First pass: mark exact matches (green)
        for i, char in enumerate(guess):
            if char == target[i]:
                feedback.append(f"{char}:GREEN")
                target_letters[i] = None  # Remove from available
            else:
                feedback.append(None)
        
        # Second pass: mark yellows
        for i, char in enumerate(guess):
            if feedback[i] is None:  # Not green
                if char in target_letters:
                    feedback[i] = f"{char}:YELLOW"
                    target_letters[target_letters.index(char)] = None
                else:
                    feedback[i] = f"{char}:GRAY"
        
        return " ".join(feedback)
    
    def _parse_guess(self, action: str) -> str:
        """
        Parse guess from XML-formatted action.
        
        Expected format:
        <think>reasoning here</think>
        <guess>WORD</guess>
        """
        # Try to extract guess from XML tags
        guess_match = re.search(r'<guess>(.*?)</guess>', action, re.IGNORECASE)
        if guess_match:
            return guess_match.group(1).strip().upper()
        
        # Fallback: if no XML tags, treat entire action as guess
        return action.strip().upper()
    
    def _mock_step(self, guess: str) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Mock step for testing without Prime Intellect backend.
        
        Uses the same logic as _simulate_prime_step for consistency.
        """
        return self._simulate_prime_step(guess)
    
    def get_state_embedding(self, state: WordleState):
        """
        Convert WordleState to numeric embedding for RL algorithms.
        
        For PPO training, we need a fixed-size numeric representation.
        This is a simple encoding strategy.
        
        Returns:
            np.ndarray: 64-dimensional state embedding
        """
        if not TORCH_AVAILABLE:
            raise ImportError("numpy is required for state embeddings. Install requirements.txt")
        
        # Features:
        # - Turn number (normalized)
        # - Number of previous guesses
        # - Vocabulary features from conversation history
        
        features = []
        
        # Turn number (0-1 normalized)
        features.append(state.turn_number / self.max_turns)
        
        # Previous guesses count (0-1 normalized)
        features.append(len(state.previous_guesses) / self.max_turns)
        
        # Game complete flag
        features.append(1.0 if state.game_complete else 0.0)
        
        # Simple text features: length of conversation
        features.append(len(state.conversation_history) / 1000.0)
        
        # Pad to fixed size (e.g., 64 dimensions)
        embedding = np.zeros(64, dtype=np.float32)
        embedding[:len(features)] = features
        
        return embedding


class WordVocabulary:
    """
    Vocabulary for Wordle words (5-letter English words).
    
    This is used to map words to action indices for discrete action space.
    """
    
    def __init__(self):
        self.words = self._load_common_words()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
    def _load_common_words(self) -> List[str]:
        """
        Load common 5-letter words for Wordle.
        
        In practice, you might want to use the actual Wordle word list.
        For now, we'll use a small set of common words.
        """
        # Common Wordle starter words and high-frequency 5-letter words
        common_words = [
            "CRANE", "SLANT", "TRACE", "SLATE", "CRATE", "LEAST", "STALE",
            "HEART", "EARTH", "DREAM", "BREAD", "STEAM", "BEAST", "FEAST",
            "PLANT", "GRAND", "STAND", "BRAND", "LIGHT", "NIGHT", "RIGHT",
            "FIGHT", "MIGHT", "SIGHT", "TIGHT", "EIGHT", "WEIGH", "NEIGH",
            "THEIR", "FIELD", "YIELD", "WIELD", "CHIEF", "GRIEF", "BRIEF",
            "AUDIO", "ADIEU", "AROSE", "IRATE", "RAISE", "ARISE", "ALERT",
            "TALES", "STEAL", "RATES", "STARE", "TEARS", "TARES", "RESAT",
            "STORE", "STONE", "NOTES", "TONES", "ONSET", "SNORE", "SPORT",
            "PORTS", "STORM", "FORMS", "WORMS", "WORDS", "LORDS", "CORDS",
            "CLEAR", "CLEAN", "CLAIM", "CLAMP", "CRAMP", "CRISP", "CLASP",
            "BLAST", "BOAST", "COAST", "ROAST", "TOAST", "BEAST", "FEAST",
            "LEARN", "EARLY", "YEARS", "TEARS", "GEARS", "BEARS", "PEARS",
            "SPEAR", "SHEAR", "SMEAR", "SWEAR", "CLEAR", "SPARE",
            "SHARE", "SNARE", "SCARE", "STARE", "STAKE", "SNAKE", "BRAKE",
            "BREAK", "BREAD", "DREAD", "TREAD", "TREND", "BLEND", "SPEND"
        ]
        
        # Remove duplicates and ensure 5 letters
        words = list(set([w.upper() for w in common_words if len(w) == 5]))
        words.sort()
        
        return words
    
    def action_to_word(self, action_idx: int) -> str:
        """Convert action index to word."""
        return self.idx_to_word.get(action_idx, "CRANE")  # Default to CRANE
    
    def word_to_action(self, word: str) -> int:
        """Convert word to action index."""
        return self.word_to_idx.get(word.upper(), 0)  # Default to first word
    
    def __len__(self) -> int:
        return len(self.words)


# Mock environment loader for when Prime Intellect is not available
def load_wordle_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    use_prime_intellect: bool = True
) -> WordleEnvironmentWrapper:
    """
    Load Wordle environment with Prime Intellect verifiers or mock fallback.
    
    Args:
        num_train_examples: Number of training episodes
        num_eval_examples: Number of evaluation episodes  
        use_prime_intellect: Whether to use Prime Intellect backend
        
    Returns:
        Wrapped Wordle environment
    """
    if use_prime_intellect:
        try:
            # This will be the actual Prime Intellect integration
            env = WordleEnvironmentWrapper(
                num_episodes=num_train_examples,
                max_turns=6
            )
            return env
        except Exception as e:
            print(f"Failed to load Prime Intellect environment: {e}")
            print("Falling back to mock environment")
    
    # Fallback to mock
    return WordleEnvironmentWrapper(
        num_episodes=num_train_examples,
        max_turns=6
    )
