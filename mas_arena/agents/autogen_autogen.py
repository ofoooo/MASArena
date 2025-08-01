from typing import Dict, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai.types.completion_usage import CompletionUsage,CompletionTokensDetails,PromptTokensDetails
import os
from dotenv import load_dotenv
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

load_dotenv()


class DDD(AgentSystem):
    def __init__(self, name: str = "ddd", config: Dict[str, Any] = None):
        """Initialize the AutoGen System"""
        super().__init__(name, config)
        self.config = config or {}

        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")

        self.max_turns = self.config.get("max_turns", 10)
        
        
    def convert_to_ai_message(self,messages):
        allmessage = []
        for msg in messages:
            usage_metadata = None
            if msg.models_usage:
                completion_tokens_details = CompletionTokensDetails(
                    accepted_prediction_tokens=0,
                    audio_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0
                )
                prompt_tokens_details = PromptTokensDetails(
                    audio_tokens=0,
                    cached_tokens=0
                )
                usage_metadata = CompletionUsage(
                    completion_tokens=msg.models_usage.completion_tokens,
                    prompt_tokens=msg.models_usage.prompt_tokens,
                    total_tokens=msg.models_usage.prompt_tokens + msg.models_usage.completion_tokens,
                    completion_tokens_details=completion_tokens_details,
                    prompt_tokens_details=prompt_tokens_details
                )
            
            ai_message = {
                'content': msg.content,
                'name': msg.source,
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': usage_metadata
            }
            allmessage.append(ai_message)
        return allmessage
    
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]: 
        
        problem_text = problem["problem"]
        # Initialize OpenAI client 
        api_key=os.getenv("OPENAI_API_KEY")
        base_url=os.getenv("OPENAI_API_BASE")
        self.model_client = OpenAIChatCompletionClient(model=self.model_name,api_key=api_key,base_url=base_url)

        self.primary = AssistantAgent(
            name="primary",
            model_client=self.model_client,
            system_message="""You are a helpful AI assistant, skilled at generating creative and accurate content."""
        )

        self.critic = AssistantAgent(
            name="critic",
            model_client=self.model_client,
            system_message="Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed."
        )

        self.group_chat = RoundRobinGroupChat(
            participants=[self.primary, self.critic],
            termination_condition=TextMentionTermination(text="APPROVE"),
            max_turns=self.max_turns
        )

        result = await self.group_chat.run(task=problem_text)

        all_messages = self.convert_to_ai_message(result.messages)
        
        primary_messages = []

        for msg in all_messages:
            if msg['name'] == 'primary':
                primary_messages.append(msg)
        final_answer = primary_messages[-1]['content']

        if("APPROVE" in all_messages[-1]["content"] ):
            all_messages = all_messages[:-1]

        await self.model_client.close()
        print(all_messages)
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

AgentSystemRegistry.register("ddd", DDD)


