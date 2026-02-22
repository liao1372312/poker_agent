from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ma_poker.agents.base import ActOutput, Agent


def _parse_rlcard_raw_obs(obs: Any) -> Dict[str, Any]:
    """Extract raw observation dict from RLCard obs."""
    if not isinstance(obs, dict):
        return {}
    raw = obs.get("raw_obs")
    return raw if isinstance(raw, dict) else {}


def _patch_rlcard_cfr_agent_get_state(cfr_agent):
    """Monkey-patch rlcard CFRAgent.get_state to fix NumPy tostring() compatibility.
    
    The rlcard library uses tostring() which was removed in NumPy 1.19+.
    This patch replaces it with tobytes() for compatibility.
    
    The original rlcard code does:
        return state['obs'].tostring(), list(state['legal_actions'].keys())
    
    We replace tostring() with tobytes() to maintain compatibility with newer NumPy.
    """
    if not hasattr(cfr_agent, 'get_state'):
        return  # Skip if method doesn't exist
    
    original_get_state = cfr_agent.get_state
    
    def patched_get_state(current_player):
        """Patched version that uses tobytes() instead of tostring()."""
        # Get state from environment (same as original)
        state = cfr_agent.env.get_state(current_player)
        
        # Fix: use tobytes() instead of tostring() for NumPy compatibility
        obs = state.get('obs')
        if obs is not None:
            if isinstance(obs, np.ndarray):
                # Use tobytes() for newer NumPy versions
                if hasattr(obs, 'tobytes'):
                    obs_bytes = obs.tobytes()
                elif hasattr(obs, 'tostring'):
                    # Fallback for older NumPy versions
                    obs_bytes = obs.tostring()
                else:
                    # Last resort: convert to bytes
                    obs_bytes = bytes(obs)
            else:
                obs_bytes = bytes(obs) if obs else b''
        else:
            obs_bytes = b''
        
        # Get legal actions (same as original)
        legal_actions = list(state.get('legal_actions', {}).keys())
        
        return obs_bytes, legal_actions
    
    cfr_agent.get_state = patched_get_state


@dataclass
class CFRAgent(Agent):
    """CFR (Counterfactual Regret Minimization) agent using rlcard library.

    This agent uses rlcard's CFRAgent for Kuhn Poker and Leduc Hold'em,
    or DeepCFRAgent for Limit Hold'em, following the implementation pattern
    from cfr_usage_example.py and CFR_Implementation_Analysis.md.
    """

    game_name: str = "leduc-holdem"  # rlcard game name: 'kuhn', 'leduc-holdem', or 'limit-holdem'
    game_params: Dict[str, Any] = None  # Optional game params
    iterations: int = 1000  # CFR iterations for training (if training on-the-fly)
    model_path: Optional[str] = None  # Path to pre-trained CFR model
    seed: Optional[int] = None
    use_deep_cfr: bool = False  # Use DeepCFR for limit-holdem
    device: Optional[str] = None  # Device for DeepCFR: 'cuda', 'cpu', or None (auto-detect)

    def __post_init__(self) -> None:
        try:
            import rlcard
        except ImportError:
            raise ImportError("rlcard is required for CFR agents. Install with: pip install rlcard")
        
        # Try to import CFR agents with error handling for encoding issues
        # rlcard.agents.__init__ may have encoding issues, so we import directly
        RLCardCFRAgent = None
        RLCardDeepCFRAgent = None
        
        try:
            # Try direct import first (may fail due to encoding issue in __init__)
            from rlcard.agents import CFRAgent as RLCardCFRAgent, DeepCFRAgent as RLCardDeepCFRAgent
        except (ImportError, UnicodeDecodeError, AttributeError) as e:
            original_error = e
            # Fallback: import modules directly to bypass __init__.py encoding issues
            try:
                import importlib.util
                import sys
                
                # Import CFRAgent directly
                cfr_agent_path = None
                for path in sys.path:
                    test_path = Path(path) / 'rlcard' / 'agents' / 'cfr_agent.py'
                    if test_path.exists():
                        cfr_agent_path = test_path
                        break
                
                if cfr_agent_path:
                    spec = importlib.util.spec_from_file_location("rlcard_agents_cfr", cfr_agent_path)
                    cfr_module = importlib.util.module_from_spec(spec)
                    # Add rlcard.agents to sys.modules to handle relative imports
                    if 'rlcard.agents' not in sys.modules:
                        import types
                        sys.modules['rlcard.agents'] = types.ModuleType('rlcard.agents')
                    spec.loader.exec_module(cfr_module)
                    RLCardCFRAgent = cfr_module.CFRAgent
                else:
                    raise ImportError("Could not find rlcard.agents.cfr_agent module")
                
                # Import DeepCFRAgent if available
                deep_cfr_path = None
                for path in sys.path:
                    test_path = Path(path) / 'rlcard' / 'agents' / 'deep_cfr_agent.py'
                    if test_path.exists():
                        deep_cfr_path = test_path
                        break
                
                if deep_cfr_path:
                    try:
                        spec = importlib.util.spec_from_file_location("rlcard_agents_deep_cfr", deep_cfr_path)
                        deep_cfr_module = importlib.util.module_from_spec(spec)
                        if 'rlcard.agents' not in sys.modules:
                            import types
                            sys.modules['rlcard.agents'] = types.ModuleType('rlcard.agents')
                        spec.loader.exec_module(deep_cfr_module)
                        RLCardDeepCFRAgent = deep_cfr_module.DeepCFRAgent
                    except Exception as deep_cfr_error:
                        print(f"[WARN] Found deep_cfr_agent.py but failed to import: {deep_cfr_error}")
                        RLCardDeepCFRAgent = None
                else:
                    # DeepCFRAgent file not found - provide diagnostic info
                    print(f"[WARN] DeepCFRAgent module not found. Searched in:")
                    for path in sys.path[:5]:  # Show first 5 paths
                        test_path = Path(path) / 'rlcard' / 'agents' / 'deep_cfr_agent.py'
                        print(f"    - {test_path} ({'EXISTS' if test_path.exists() else 'NOT FOUND'})")
                    try:
                        import rlcard
                        print(f"[INFO] rlcard version: {rlcard.__version__ if hasattr(rlcard, '__version__') else 'unknown'}")
                        print(f"[INFO] rlcard location: {rlcard.__file__ if hasattr(rlcard, '__file__') else 'unknown'}")
                    except:
                        pass
                    RLCardDeepCFRAgent = None
                # Note: DeepCFRAgent may not be available in all rlcard versions
                
            except Exception as e2:
                raise ImportError(
                    f"Failed to import rlcard CFR agents. "
                    f"Original error: {original_error}. "
                    f"Fallback error: {e2}. "
                    f"This may be due to encoding issues in rlcard.agents.__init__.py. "
                    f"Try: pip install --upgrade rlcard"
                )
        
        # Verify that required agents are imported
        if RLCardCFRAgent is None:
            raise ImportError("Failed to import CFRAgent from rlcard.agents")
        
        # Determine game name based on environment
        # Map common names to rlcard game names
        # Note: rlcard supports: 'blackjack', 'doudizhu', 'limit-holdem', 'no-limit-holdem', 
        #       'leduc-holdem', 'uno', 'mahjong', 'gin-rummy', 'bridge'
        #       It does NOT support 'kuhn' directly, but kuhn models can be used with leduc-holdem
        game_name_mapping = {
            "texas_holdem": "limit-holdem",  # Default to limit-holdem for Texas Hold'em
            "limit-holdem": "limit-holdem",
            "no-limit-holdem": "limit-holdem",  # rlcard doesn't have no-limit, use limit
            "leduc-holdem": "leduc-holdem",
            "kuhn": "leduc-holdem",  # rlcard doesn't have kuhn, use leduc-holdem (similar game)
        }
        
        rlcard_game_name = game_name_mapping.get(self.game_name.lower(), "leduc-holdem")
        
        # Determine if we should use DeepCFR
        if rlcard_game_name == "limit-holdem":
            self.use_deep_cfr = True
            self.rlcard_game_name = "limit-holdem"
            print(f"[INFO] CFR: Game '{self.game_name}' mapped to 'limit-holdem', will use DeepCFR (supports GPU)")
        else:
            self.use_deep_cfr = False
            self.rlcard_game_name = rlcard_game_name
            print(f"[INFO] CFR: Game '{self.game_name}' mapped to '{rlcard_game_name}', will use Standard CFR (CPU only)")
        
        # Create rlcard environment for CFR training/usage
        config = {'seed': self.seed if self.seed is not None else 0, 'allow_step_back': True}
        self._rlcard_env = rlcard.make(self.rlcard_game_name, config=config)
        
        # Initialize CFR agent
        if self.use_deep_cfr:
            # Use DeepCFR for limit-holdem
            if RLCardDeepCFRAgent is None:
                # Try to use our GPU-accelerated DeepCFR implementation
                try:
                    from ma_poker.agents.gpu_deep_cfr import GPUDeepCFRAgent
                    print("[INFO] =========================================")
                    print("[INFO] Using GPU-accelerated DeepCFR implementation")
                    print(f"[INFO] This implementation uses PyTorch neural networks for GPU acceleration")
                    print(f"[INFO] =========================================")
                    
                    # Use our GPU DeepCFR implementation
                    model_path = self.model_path or f'./pretrained_models/{self.rlcard_game_name}_gpu_deep_cfr_result/model.pt'
                    self._cfr_agent = GPUDeepCFRAgent(
                        game_name=self.rlcard_game_name,
                        iterations=self.iterations,
                        model_path=model_path,
                        seed=self.seed,
                        device=self.device,
                    )
                    # Keep use_deep_cfr = True to use DeepCFR training logic
                except ImportError as gpu_cfr_error:
                    # Fallback to standard CFR if GPU DeepCFR is not available
                    print("[WARN] =========================================")
                    print("[WARN] DeepCFRAgent is NOT available in rlcard.")
                    print(f"[WARN] GPU-accelerated DeepCFR also failed: {gpu_cfr_error}")
                    print(f"[WARN] rlcard does not include DeepCFRAgent in its standard distribution.")
                    print(f"[WARN] Falling back to standard CFR for {self.rlcard_game_name}.")
                    print(f"[WARN] Note: Standard CFR may not scale well to limit-holdem.")
                    print(f"[WARN] Standard CFR does NOT support GPU - it uses CPU only.")
                    print(f"[WARN] =========================================")
                    print(f"[WARN] IMPORTANT: DeepCFRAgent is not part of standard rlcard.")
                    print(f"[WARN] The rlcard library (v1.2.0+) does not include DeepCFRAgent.")
                    print(f"[WARN] To use GPU-accelerated CFR, you would need to:")
                    print(f"[WARN]   1. Install PyTorch: pip install torch")
                    print(f"[WARN]   2. Use GPU DeepCFR (requires PyTorch)")
                    print(f"[WARN]   3. Use standard CFR with fewer iterations for faster training")
                    print(f"[INFO] CFR: Using Standard CFR (CPU only, no GPU support)")
                    # Use standard CFR instead
                    self.use_deep_cfr = False
                    model_path = self.model_path or f'./pretrained_models/{self.rlcard_game_name}_cfr_result/cfr_model'
                    self._cfr_agent = RLCardCFRAgent(
                        env=self._rlcard_env,
                        model_path=model_path
                    )
                    # Apply compatibility patch for NumPy tostring() issue
                    _patch_rlcard_cfr_agent_get_state(self._cfr_agent)
            else:
                # Use DeepCFR as intended
                try:
                    import torch
                    # Determine device: use specified device, or auto-detect GPU availability
                    if self.device:
                        device = torch.device(self.device)
                        device_source = "user-specified"
                    else:
                        # Auto-detect: use CUDA if available, otherwise CPU
                        if torch.cuda.is_available():
                            device = torch.device('cuda')
                            device_source = "auto-detected (GPU available)"
                        else:
                            device = torch.device('cpu')
                            device_source = "auto-detected (GPU not available)"
                    
                    # Print detailed device information
                    if device.type == 'cuda':
                        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                        gpu_count = torch.cuda.device_count()
                        print(f"[INFO] DeepCFR: Using GPU for training")
                        print(f"    Device: {device}")
                        print(f"    GPU Name: {gpu_name}")
                        print(f"    GPU Count: {gpu_count}")
                        print(f"    Source: {device_source}")
                    else:
                        print(f"[INFO] DeepCFR: Using CPU for training")
                        print(f"    Device: {device}")
                        print(f"    Source: {device_source}")
                        if torch.cuda.is_available():
                            print(f"    [NOTE] GPU is available but not being used. Set device='cuda' to use GPU.")
                except ImportError:
                    device = None
                    print("[WARN] DeepCFR: PyTorch not available, cannot use GPU")
                
                model_path = self.model_path or f'./pretrained_models/{self.rlcard_game_name}_deep_cfr_result/cfr_model'
                self._cfr_agent = RLCardDeepCFRAgent(
                    env=self._rlcard_env,
                    model_path=model_path,
                    device=device
                )
            
            # Initialize training flag
            self._trained = False
            
            # Try to load model/checkpoint if exists
            if self.use_deep_cfr:
                # Our GPUDeepCFRAgent loads model.pt in its __init__; no rlcard checkpoint path
                if hasattr(self._cfr_agent, 'save_model'):
                    self._trained = getattr(self._cfr_agent, '_trained', False)
                else:
                    # rlcard DeepCFR: load checkpoint_deep_cfr.pt
                    checkpoint_path = f"pretrained_models/{self.rlcard_game_name}_deep_cfr_result/checkpoint_deep_cfr.pt"
                    if self.model_path:
                        checkpoint_path = str(Path(self.model_path).parent / "checkpoint_deep_cfr.pt")
                    try:
                        checkpoint_file = Path(checkpoint_path)
                        if checkpoint_file.exists():
                            self._cfr_agent.load_checkpoint(str(checkpoint_file))
                            print(f"[OK] DeepCFR: Loaded checkpoint from {checkpoint_path}")
                            self._trained = True
                        else:
                            model_dir = Path(self.model_path).parent if self.model_path else Path(f"pretrained_models/{self.rlcard_game_name}_deep_cfr_result")
                            if model_dir.exists():
                                checkpoint_files = list(model_dir.glob("*.pt"))
                                if checkpoint_files:
                                    try:
                                        self._cfr_agent.load_checkpoint(str(checkpoint_files[0]))
                                        print(f"[OK] DeepCFR: Loaded checkpoint from {checkpoint_files[0]}")
                                        self._trained = True
                                    except Exception as e2:
                                        print(f"[WARN]  DeepCFR: Failed to load checkpoint {checkpoint_files[0]}: {e2}")
                                else:
                                    print(f"[WARN]  DeepCFR: No checkpoint found at {checkpoint_path} or in {model_dir}")
                            else:
                                print(f"[WARN]  DeepCFR: Checkpoint not found at {checkpoint_path}, will train from scratch")
                    except Exception as e:
                        print(f"[WARN]  DeepCFR: Failed to load checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                # Standard CFR uses model files (pkl) - fallback case
                try:
                    model_dir = Path(self._cfr_agent.model_path)
                    if model_dir.exists() and model_dir.is_dir():
                        # Check if model files exist
                        required_files = ['regrets.pkl', 'average_policy.pkl', 'policy.pkl']
                        has_all_files = all((model_dir / f).exists() for f in required_files)
                        
                        if has_all_files:
                            self._cfr_agent.load()
                            print(f"[OK] CFR: Loaded CFR model from {model_dir}")
                            self._trained = True  # Model loaded, no need to train
                        else:
                            print(f"[WARN]  CFR: Model directory exists but missing required files at {model_dir}")
                            print(f"    Missing files: {[f for f in required_files if not (model_dir / f).exists()]}")
                    else:
                        print(f"[WARN]  CFR: CFR model not found at {model_dir}, will train from scratch")
                except Exception as e:
                    print(f"[WARN]  CFR: Failed to load CFR model: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Use standard CFR for kuhn and leduc-holdem
            print(f"[INFO] CFR: Using Standard CFR for {self.rlcard_game_name} (CPU only, no GPU support)")
            model_path = self.model_path or f'./pretrained_models/{self.rlcard_game_name}_cfr_result/cfr_model'
            
            # Initialize training flag
            self._trained = False
            
            # Special handling for leduc-holdem: try rlcard built-in model first
            if self.rlcard_game_name == "leduc-holdem" and not self.model_path:
                try:
                    from rlcard import models
                    print(f"[CFR] Attempting to load rlcard built-in model for leduc-holdem...")
                    builtin_model = models.load('leduc-holdem-cfr')
                    self._cfr_agent = builtin_model.agents[0]
                    # Apply compatibility patch for NumPy tostring() issue
                    _patch_rlcard_cfr_agent_get_state(self._cfr_agent)
                    print(f"[CFR] Loaded rlcard built-in leduc-holdem-cfr model")
                    self._trained = True  # Built-in model is already trained
                    self._training_iterations = 0
                    return
                except Exception as e:
                    print(f"[CFR] Failed to load rlcard built-in model: {e}")
                    print(f"    Will try local model path: {model_path}")
            
            # Initialize CFR agent with model path
            self._cfr_agent = RLCardCFRAgent(
                env=self._rlcard_env,
                model_path=model_path
            )
            # Apply compatibility patch for NumPy tostring() issue
            _patch_rlcard_cfr_agent_get_state(self._cfr_agent)
            
            # Try to load model if exists
            try:
                model_dir = Path(model_path)
                if model_dir.exists() and model_dir.is_dir():
                    # Check if model files exist
                    required_files = ['regrets.pkl', 'average_policy.pkl', 'policy.pkl']
                    has_all_files = all((model_dir / f).exists() for f in required_files)
                    
                    if has_all_files:
                        self._cfr_agent.load()
                        print(f"[OK] CFR: Loaded CFR model from {model_path}")
                        self._trained = True  # Model loaded, no need to train
                    else:
                        print(f"[WARN]  CFR: Model directory exists but missing required files at {model_path}")
                        print(f"    Missing files: {[f for f in required_files if not (model_dir / f).exists()]}")
                else:
                    print(f"[WARN]  CFR: CFR model not found at {model_path}, will train from scratch")
            except Exception as e:
                print(f"[WARN]  CFR: Failed to load CFR model: {e}")
                import traceback
                traceback.print_exc()
        
        # Initialize training iterations counter
        if not hasattr(self, '_training_iterations'):
            self._training_iterations = 0

    def reset(self) -> None:
        """Reset agent state."""
        # CFR agents don't need per-episode reset, but we can reset training flag
        pass

    def _train_cfr(self, num_iterations: Optional[int] = None) -> None:
        """Train CFR agent if not already trained."""
        if self._trained:
            return
        
        iterations = num_iterations or self.iterations
        
        # Print device information for DeepCFR
        if self.use_deep_cfr:
            try:
                import torch
                if hasattr(self, '_cfr_agent') and hasattr(self._cfr_agent, 'device'):
                    device = self._cfr_agent.device if hasattr(self._cfr_agent, 'device') else None
                    if device:
                        if device.type == 'cuda':
                            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                            print(f"[INFO] CFR: Training {self.rlcard_game_name} DeepCFR agent for {iterations} iterations...")
                            print(f"    Training Device: GPU ({gpu_name})")
                        else:
                            print(f"[INFO] CFR: Training {self.rlcard_game_name} DeepCFR agent for {iterations} iterations...")
                            print(f"    Training Device: CPU")
                    else:
                        print(f"[INFO] CFR: Training {self.rlcard_game_name} DeepCFR agent for {iterations} iterations...")
                else:
                    print(f"[INFO] CFR: Training {self.rlcard_game_name} DeepCFR agent for {iterations} iterations...")
            except:
                print(f"[INFO] CFR: Training {self.rlcard_game_name} DeepCFR agent for {iterations} iterations...")
        else:
            print(f"[INFO] CFR: Training {self.rlcard_game_name} CFR agent for {iterations} iterations...")
            print(f"    Training Device: CPU (Standard CFR uses CPU)")
        
        print(f"    This may take a while. Model will be saved after training.")
        
        # Check if it's our GPUDeepCFRAgent (which trains all iterations at once)
        # vs rlcard's CFRAgent (which trains one iteration per call)
        if self.use_deep_cfr and hasattr(self._cfr_agent, 'save_model'):
            # Our GPUDeepCFRAgent: train all iterations at once
            self._cfr_agent.train(num_iterations=iterations)
            self._training_iterations = iterations
        else:
            # Standard CFR or rlcard's DeepCFR: train one iteration per call
            # Determine progress reporting frequency based on total iterations
            if iterations <= 50:
                progress_interval = 10  # Every 10 iterations for small runs
            elif iterations <= 200:
                progress_interval = 20  # Every 20 iterations for medium runs
            else:
                progress_interval = 100  # Every 100 iterations for large runs
            
            import time
            start_time = time.time()
            
            for i in range(iterations):
                self._cfr_agent.train()
                self._training_iterations += 1
                
                # Report progress at intervals
                if (i + 1) % progress_interval == 0 or (i + 1) == iterations:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (iterations - (i + 1)) / rate if rate > 0 else 0
                    progress_pct = 100 * (i + 1) / iterations
                    print(f"  CFR training: {i + 1}/{iterations} ({progress_pct:.1f}%) | "
                          f"Rate: {rate:.2f} it/s | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Remaining: {remaining:.1f}s")
        
        # Save after training
        try:
            if self.use_deep_cfr:
                # Check if it's our GPU DeepCFR or rlcard's DeepCFR
                if hasattr(self._cfr_agent, 'save_model'):
                    # Our GPU DeepCFR implementation
                    model_path = self.model_path or f'./pretrained_models/{self.rlcard_game_name}_gpu_deep_cfr_result/model.pt'
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    self._cfr_agent.save_model(model_path)
                    print(f"[OK] CFR: Saved trained GPU DeepCFR model to {model_path}")
                elif hasattr(self._cfr_agent, 'save_checkpoint'):
                    # rlcard's DeepCFR
                    checkpoint_path = f"pretrained_models/{self.rlcard_game_name}_deep_cfr_result/checkpoint_deep_cfr.pt"
                    if self.model_path:
                        checkpoint_path = str(Path(self.model_path).parent / "checkpoint_deep_cfr.pt")
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    self._cfr_agent.save_checkpoint(checkpoint_path)
                    print(f"[OK] CFR: Saved trained DeepCFR checkpoint to {checkpoint_path}")
                else:
                    print(f"[WARN] CFR: Unknown DeepCFR implementation, cannot save model")
            else:
                model_path = Path(self._cfr_agent.model_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self._cfr_agent.save()
                print(f"[OK] CFR: Saved trained CFR model to {model_path}")
        except Exception as e:
            print(f"[WARN]  CFR: Failed to save model: {e}")
            import traceback
            traceback.print_exc()
        
        self._trained = True

    def _expected_obs_len(self) -> int:
        """Expected obs vector length for this CFR/DeepCFR model (limit-holdem=72)."""
        sh = self._rlcard_env.state_shape[0]
        if isinstance(sh, (list, tuple)):
            return int(np.prod(sh)) if sh else 72
        return int(sh) if sh is not None else 72

    def _convert_rlcard_obs_to_rlcard_state(self, obs: Any, legal_actions: Optional[List[int]] = None) -> Dict[str, Any]:
        """Convert our RLCard observation format to rlcard CFRAgent's expected state format.
        
        Prefer passed-in legal_actions (from act(obs, legal_actions)) so we stay consistent
        with what the env/runner provided; fall back to obs['legal_actions'] when not given.
        This agent's model is for limit-holdem (72-dim obs, 4 actions). If obs is from
        no-limit (e.g. 54-dim) or missing, we resize/pad to expected length and warn.
        """
        expected_len = self._expected_obs_len()
        obs_vec = obs.get("obs") if isinstance(obs, dict) else None
        if obs_vec is None:
            if not getattr(self, "_warned_obs_missing", False):
                print("[WARN] CFR/DeepCFR: obs['obs'] missing — using zeros. Use env_id: limit-holdem when evaluating DeepCFR.")
                self._warned_obs_missing = True
            obs_vec = np.zeros(expected_len, dtype=np.float32)
        else:
            if not isinstance(obs_vec, np.ndarray):
                obs_vec = np.array(obs_vec, dtype=np.float32)
            n = len(obs_vec)
            if n != expected_len:
                if not getattr(self, "_warned_obs_len", False):
                    print(f"[WARN] CFR/DeepCFR: obs length {n} != model expected {expected_len} (limit-holdem). "
                          "Resizing. Use env_id: limit-holdem when evaluating DeepCFR.")
                    self._warned_obs_len = True
                if n < expected_len:
                    obs_vec = np.concatenate([obs_vec, np.zeros(expected_len - n, dtype=np.float32)])
                else:
                    obs_vec = obs_vec[:expected_len].astype(np.float32)
        
        if legal_actions is not None:
            legal_actions_dict = {int(a): None for a in legal_actions}
        else:
            legal_actions_dict = {}
            legal_action_ids = obs.get("legal_actions", []) if isinstance(obs, dict) else []
            if isinstance(legal_action_ids, list):
                for action_id in legal_action_ids:
                    legal_actions_dict[int(action_id)] = None
            elif isinstance(legal_action_ids, dict):
                legal_actions_dict = {int(k): v for k, v in legal_action_ids.items()}
        
        state = {'obs': obs_vec, 'legal_actions': legal_actions_dict}
        return state

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        """Select action using CFR strategy."""
        if not legal_actions:
            raise ValueError("No legal actions provided.")
        
        # Train CFR if not trained yet
        if not self._trained:
            self._train_cfr()
        
        # Convert observation to rlcard state format (use passed-in legal_actions for consistency)
        try:
            state = self._convert_rlcard_obs_to_rlcard_state(obs, legal_actions)
        except Exception as e:
            # Fallback to random if conversion fails
            import random
            rng = random.Random(self.seed)
            action = rng.choice(legal_actions)
            return ActOutput(action=int(action), info={"policy": "cfr_fallback_random", "error": str(e)})
        
        # Get action from CFR agent
        try:
            # Handle both rlcard's CFRAgent and our GPUDeepCFRAgent
            if hasattr(self._cfr_agent, 'eval_step'):
                action_id, action_probs = self._cfr_agent.eval_step(state)
            elif hasattr(self._cfr_agent, 'act'):
                # Our GPUDeepCFRAgent uses act() method
                act_output = self._cfr_agent.act(obs, legal_actions)
                action_id = act_output.action
                action_probs = act_output.info.get("action_probs", {})
            else:
                raise AttributeError("CFR agent does not have eval_step or act method")
            
            # Ensure action_id is in legal_actions
            if action_id not in legal_actions:
                # Find closest legal action or use first legal action
                if legal_actions:
                    action_id = legal_actions[0]
                else:
                    raise ValueError("No legal actions available")
            
            # Get action probabilities for legal actions
            legal_action_probs = {}
            if isinstance(action_probs, dict):
                for aid in legal_actions:
                    legal_action_probs[aid] = float(action_probs.get(aid, 0.0))
            elif isinstance(action_probs, (list, np.ndarray)):
                for i, aid in enumerate(legal_actions):
                    if i < len(action_probs):
                        legal_action_probs[aid] = float(action_probs[i])
            
            return ActOutput(
                action=int(action_id),
                info={
                    "policy": "cfr" if not self.use_deep_cfr else "deep_cfr",
                    "action_probs": legal_action_probs,
                    "game": self.rlcard_game_name,
                }
            )
        except Exception as e:
            # Fallback to random if CFR fails
            import random
            rng = random.Random(self.seed)
            action = rng.choice(legal_actions)
            return ActOutput(
                action=int(action),
                info={"policy": "cfr_fallback_random", "error": str(e)}
            )


@dataclass
class DeepCFRAgent(Agent):
    """DeepCFR agent using rlcard library.

    DeepCFR uses neural networks to approximate CFR, making it scalable to larger games.
    This is specifically for limit-holdem using rlcard's DeepCFRAgent.
    """

    game_name: str = "limit-holdem"  # rlcard game name (should be 'limit-holdem')
    game_params: Dict[str, Any] = None
    model_path: Optional[str] = None
    seed: Optional[int] = None
    iterations: int = 1000  # Training iterations
    device: Optional[str] = None  # Device for DeepCFR: 'cuda', 'cpu', or None (auto-detect)

    def __post_init__(self) -> None:
        try:
            import rlcard
        except ImportError:
            raise ImportError("rlcard is required for DeepCFR. Install with: pip install rlcard")
        
        # Try to import DeepCFR agent with error handling for encoding issues
        RLCardDeepCFRAgent = None
        
        try:
            # Try direct import first (may fail due to encoding issue in __init__)
            from rlcard.agents import DeepCFRAgent as RLCardDeepCFRAgent
        except (ImportError, UnicodeDecodeError, AttributeError) as e:
            # Fallback: import module directly to bypass __init__.py encoding issues
            try:
                import importlib.util
                import sys
                from pathlib import Path
                
                # Find rlcard installation path
                rlcard_path = None
                for path in sys.path:
                    test_path = Path(path) / 'rlcard' / 'agents' / 'deep_cfr_agent.py'
                    if test_path.exists():
                        rlcard_path = Path(path) / 'rlcard' / 'agents'
                        break
                
                if rlcard_path:
                    deep_cfr_file = rlcard_path / 'deep_cfr_agent.py'
                    spec = importlib.util.spec_from_file_location("rlcard_agents_deep_cfr", deep_cfr_file)
                    deep_cfr_module = importlib.util.module_from_spec(spec)
                    # Add rlcard.agents to sys.modules to handle relative imports
                    if 'rlcard.agents' not in sys.modules:
                        import types
                        sys.modules['rlcard.agents'] = types.ModuleType('rlcard.agents')
                    spec.loader.exec_module(deep_cfr_module)
                    RLCardDeepCFRAgent = deep_cfr_module.DeepCFRAgent
                else:
                    raise ImportError("Could not find rlcard.agents.deep_cfr_agent module")
                    
            except Exception as e2:
                raise ImportError(
                    f"Failed to import rlcard DeepCFRAgent. "
                    f"Original error: {e}. "
                    f"Fallback error: {e2}. "
                    f"This may be due to encoding issues in rlcard.agents.__init__.py. "
                    f"Try: pip install --upgrade rlcard"
                )
        
        try:
            import torch
            # Determine device: use specified device, or auto-detect GPU availability
            if self.device:
                self._device = torch.device(self.device)
                device_source = "user-specified"
            else:
                # Auto-detect: use CUDA if available, otherwise CPU
                if torch.cuda.is_available():
                    self._device = torch.device('cuda')
                    device_source = "auto-detected (GPU available)"
                else:
                    self._device = torch.device('cpu')
                    device_source = "auto-detected (GPU not available)"
            
            # Print detailed device information
            if self._device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                gpu_count = torch.cuda.device_count()
                print(f"[INFO] DeepCFR: Using GPU for training")
                print(f"    Device: {self._device}")
                print(f"    GPU Name: {gpu_name}")
                print(f"    GPU Count: {gpu_count}")
                print(f"    Source: {device_source}")
            else:
                print(f"[INFO] DeepCFR: Using CPU for training")
                print(f"    Device: {self._device}")
                print(f"    Source: {device_source}")
                if torch.cuda.is_available():
                    print(f"    [NOTE] GPU is available but not being used. Set device='cuda' to use GPU.")
        except ImportError:
            self._device = None
            print("[WARN]  DeepCFR: PyTorch not available, some features may not work")
        
        # Create rlcard environment
        config = {'seed': self.seed if self.seed is not None else 0, 'allow_step_back': True}
        self._rlcard_env = rlcard.make("limit-holdem", config=config)
        
        # Initialize DeepCFR agent
        model_path = self.model_path or f'./pretrained_models/limit-holdem_deep_cfr_result/cfr_model'
        self._cfr_agent = RLCardDeepCFRAgent(
            env=self._rlcard_env,
            model_path=model_path,
            device=self._device
        )
        
        # Try to load checkpoint if exists
        checkpoint_path = f"pretrained_models/limit-holdem_deep_cfr_result/checkpoint_deep_cfr.pt"
        if self.model_path:
            checkpoint_path = str(Path(self.model_path).parent / "checkpoint_deep_cfr.pt")
        
        try:
            if Path(checkpoint_path).exists():
                self._cfr_agent.load_checkpoint(checkpoint_path)
                print(f"[OK] DeepCFR: Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"[WARN]  DeepCFR: Checkpoint not found at {checkpoint_path}, will train from scratch")
        except Exception as e:
            print(f"[WARN]  DeepCFR: Failed to load checkpoint: {e}")
        
        self._trained = False
        self._training_iterations = 0

    def reset(self) -> None:
        """Reset agent state."""
        pass

    def _train_deep_cfr(self, num_iterations: Optional[int] = None) -> None:
        """Train DeepCFR agent if not already trained."""
        if self._trained:
            return
        
        iterations = num_iterations or self.iterations
        
        # Print device information
        try:
            import torch
            if self._device and self._device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                print(f"[INFO] DeepCFR: Training limit-holdem DeepCFR agent for {iterations} iterations...")
                print(f"    Training Device: GPU ({gpu_name})")
            else:
                print(f"[INFO] DeepCFR: Training limit-holdem DeepCFR agent for {iterations} iterations...")
                print(f"    Training Device: CPU")
        except:
            print(f"[INFO] DeepCFR: Training limit-holdem DeepCFR agent for {iterations} iterations...")
        
        # Determine progress reporting frequency based on total iterations
        if iterations <= 50:
            progress_interval = 10  # Every 10 iterations for small runs
        elif iterations <= 200:
            progress_interval = 20  # Every 20 iterations for medium runs
        else:
            progress_interval = 100  # Every 100 iterations for large runs
        
        import time
        start_time = time.time()
        
        for i in range(iterations):
            self._cfr_agent.train()
            self._training_iterations += 1
            
            # Report progress at intervals
            if (i + 1) % progress_interval == 0 or (i + 1) == iterations:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (iterations - (i + 1)) / rate if rate > 0 else 0
                progress_pct = 100 * (i + 1) / iterations
                print(f"  DeepCFR training: {i + 1}/{iterations} ({progress_pct:.1f}%) | "
                      f"Rate: {rate:.2f} it/s | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Remaining: {remaining:.1f}s")
        
        # Save after training
        try:
            checkpoint_path = f"pretrained_models/limit-holdem_deep_cfr_result/checkpoint_deep_cfr.pt"
            if self.model_path:
                checkpoint_path = str(Path(self.model_path).parent / "checkpoint_deep_cfr.pt")
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            self._cfr_agent.save_checkpoint(checkpoint_path)
            print(f"[OK] DeepCFR: Saved trained checkpoint")
        except Exception as e:
            print(f"[WARN]  DeepCFR: Failed to save checkpoint: {e}")
        
        self._trained = True

    def _convert_rlcard_obs_to_rlcard_state(self, obs: Any, legal_actions: Optional[List[int]] = None) -> Dict[str, Any]:
        """Convert RLCard observation to state format. Prefer passed-in legal_actions when provided."""
        obs_vec = obs.get("obs") if isinstance(obs, dict) else None
        if obs_vec is None:
            obs_vec = np.zeros(self._rlcard_env.state_shape[0], dtype=np.float32)
        if not isinstance(obs_vec, np.ndarray):
            obs_vec = np.array(obs_vec, dtype=np.float32)
        if legal_actions is not None:
            legal_actions_dict = {int(a): None for a in legal_actions}
        else:
            legal_actions_dict = {}
            legal_action_ids = obs.get("legal_actions", []) if isinstance(obs, dict) else []
            if isinstance(legal_action_ids, list):
                for action_id in legal_action_ids:
                    legal_actions_dict[int(action_id)] = None
            elif isinstance(legal_action_ids, dict):
                legal_actions_dict = {int(k): v for k, v in legal_action_ids.items()}
        return {'obs': obs_vec, 'legal_actions': legal_actions_dict}

    def act(self, obs: Any, legal_actions: List[int]) -> ActOutput:
        if not legal_actions:
            raise ValueError("No legal actions provided.")

        # Train DeepCFR if not trained yet
        if not self._trained:
            self._train_deep_cfr()

        # Convert observation to rlcard state format (use passed-in legal_actions for consistency)
        try:
            state = self._convert_rlcard_obs_to_rlcard_state(obs, legal_actions)
        except Exception as e:
            import random
            rng = random.Random(self.seed)
            action = rng.choice(legal_actions)
            return ActOutput(action=int(action), info={"policy": "deepcfr_fallback_random", "error": str(e)})

        # Get action from DeepCFR agent
        try:
            action_id, action_probs = self._cfr_agent.eval_step(state)
            
            # Ensure action_id is in legal_actions
            if action_id not in legal_actions:
                if legal_actions:
                    action_id = legal_actions[0]
                else:
                    raise ValueError("No legal actions available")
            
            # Get action probabilities for legal actions
            legal_action_probs = {}
            if isinstance(action_probs, dict):
                for aid in legal_actions:
                    legal_action_probs[aid] = float(action_probs.get(aid, 0.0))
            elif isinstance(action_probs, (list, np.ndarray)):
                for i, aid in enumerate(legal_actions):
                    if i < len(action_probs):
                        legal_action_probs[aid] = float(action_probs[i])
            
            return ActOutput(
                action=int(action_id),
                info={
                    "policy": "deep_cfr",
                    "action_probs": legal_action_probs,
                    "game": "limit-holdem",
                }
            )
        except Exception as e:
            import random
            rng = random.Random(self.seed)
            action = rng.choice(legal_actions)
            return ActOutput(
                action=int(action),
                info={"policy": "deepcfr_fallback_random", "error": str(e)}
            )
