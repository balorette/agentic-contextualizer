# Continuing Build

So as we continue this build it will be good to pull in the token counts for building the context maybe even offer the ability to specifiy the model the context will be built for. 

```python
class ModelTokens:
    def __init__(self):
        self.models = {}
    
    def get_model_contexts(self):
        """Reach out and pull context windows from models"""
        pass

    def context_windows(self):
        """ Retrun model windows"""
        pass

    def get_context_windo(self, model: str) -> int:
        """ Return the context window size of a specified model """
        pass
```

Use TikToken to then estimate the context sizes in order to keep things with in desired parameters. 