from vizdoom import DoomGame


class DoomEnvironment:
    def __init__(self, config_path, frame_skip=4):
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.init()
        self.frame_skip = frame_skip

    def reset(self):
        self.game.new_episode()
        return self._process_state(self.game.get_state())

    def step(self, action):
        reward = self.game.make_action(action, self.frame_skip)
        done = self.game.is_episode_finished()
        state = None if done else self._process_state(self.game.get_state())
        return state, reward, done

    def close(self):
        self.game.close()

    @staticmethod
    def _process_state(state):
        if state is not None:
            frame = state.screen_buffer  # Shape: [C, H, W]
            return frame
        return None
