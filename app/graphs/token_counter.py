from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenCountCallback(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a pipeline run.

    Works with OpenAI, Gemini (via usage_metadata on AIMessage) and Ollama
    completion models (via eval_count / prompt_eval_count in llm_output).
    Attach to a LangGraph run via config={"callbacks": [cb]} — LangGraph
    propagates the callback to every nested LLM call automatically.
    """

    def __init__(self):
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.llm_calls: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.llm_calls += 1
        for gen_list in response.generations:
            for gen in gen_list:
                # Chat model path: AIMessage carries usage_metadata
                msg = getattr(gen, "message", None)
                if msg is not None:
                    meta = getattr(msg, "usage_metadata", None) or {}
                    self.input_tokens += meta.get("input_tokens", 0)
                    self.output_tokens += meta.get("output_tokens", 0)
                    return
        # Completion LLM path (OllamaLLM): token counts live in llm_output
        llm_out = response.llm_output or {}
        self.input_tokens += llm_out.get("prompt_eval_count", 0)
        self.output_tokens += llm_out.get("eval_count", 0)
