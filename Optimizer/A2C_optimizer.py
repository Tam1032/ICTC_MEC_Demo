from stable_baselines3 import A2C

class A2C_Optimizer:
    def __init__(self, env):
        self.env = env
        self.model = A2C("MultiInputPolicy", self.env, verbose=1)
        self.save_path = None

    def train(self, total_timesteps=30000, 
              model_save_path="A2C_mec_offload", 
              save_model=True):
        # Train the agent
        self.model.learn(total_timesteps=total_timesteps)
        # Save the trained model
        if save_model:
            self.save_path = model_save_path
            self.model.save(model_save_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained A2C model.
        """
        self.save_path = model_path
        self.model = A2C.load(model_path, env=self.env)
        
    def predict(self, obs, model_path="A2C_mec_offload"):
        action, _states = self.model.predict(obs)
        return action

# Example usage:
# train_and_test_a2c()