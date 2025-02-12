from vizdoom import DoomGame


# Reprezentacja środowiska gry i obsługa środowiska dostarczonego przez bibliotekę VizDoom
class DoomEnvironment:
    def __init__(self, config_path, action_map, frame_skip=4):
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.init()
        self.frame_skip = frame_skip
        self.action_map = action_map

    def reset(self):
        self.game.new_episode()
        return self._process_state(self.game.get_state())

    def step(self, action):
        reward = self.game.make_action(self.action_map[action[0]], self.frame_skip)
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
