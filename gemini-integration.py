"""Gemini Assistant Agent."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from llama_index.core.agent.function_calling.step import (
    build_error_tool_output,
    build_missing_tool_message,
    get_function_by_name,
)
from llama_index.core.agent.types import BaseAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
    ChatResponseMode,
    StreamingAgentChatResponse,
)
from llama_index.core.tools import BaseTool, ToolOutput, adapt_to_async_tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def from_gemini_message(message: Any) -> ChatMessage:
    """Convert Gemini message to ChatMessage."""
    role = MessageRole.USER if message.role == "user" else MessageRole.ASSISTANT
    
    # Extract content from the message
    if hasattr(message, "parts"):
        content = " ".join([part.text for part in message.parts if hasattr(part, "text")])
    else:
        content = message.content
        
    return ChatMessage(
        role=role,
        content=content,
        additional_kwargs={
            "gemini_message": message,
        },
    )


def from_gemini_messages(messages: List[Any]) -> List[ChatMessage]:
    """Convert list of Gemini messages to ChatMessages."""
    return [from_gemini_message(message) for message in messages]


def to_gemini_message(message: ChatMessage) -> Dict[str, Any]:
    """Convert ChatMessage to Gemini message format."""
    role = "user" if message.role == MessageRole.USER else "model"
    
    # Handle function messages
    if message.role == MessageRole.FUNCTION:
        # Gemini supports function responses as content
        return {
            "role": "function",
            "parts": [{"text": message.content}],
            "name": message.additional_kwargs.get("name", "function")
        }
    
    return {
        "role": role,
        "parts": [{"text": message.content}]
    }


def call_function(
    tools: List[BaseTool], fn_obj: Any, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call a function and return the output as a string."""
    name = fn_obj.get("name")
    arguments_str = fn_obj.get("arguments", "{}")

    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")

    tool = get_function_by_name(tools, name)
    if tool is not None:
        argument_dict = json.loads(arguments_str)
        output = tool(**argument_dict)

        if verbose:
            print(f"Got output: {output!s}")
            print("========================")
    else:
        err_msg = build_missing_tool_message(name)
        output = build_error_tool_output(name, arguments_str, err_msg)

        if verbose:
            print(err_msg)
            print("========================")

    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.FUNCTION,
            additional_kwargs={
                "name": name,
            },
        ),
        output,
    )


async def acall_function(
    tools: List[BaseTool], fn_obj: Any, verbose: bool = False
) -> Tuple[ChatMessage, ToolOutput]:
    """Call an async function and return the output as a string."""
    name = fn_obj.get("name")
    arguments_str = fn_obj.get("arguments", "{}")

    if verbose:
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")

    tool = get_function_by_name(tools, name)
    if tool is not None:
        argument_dict = json.loads(arguments_str)
        tool = adapt_to_async_tool(tool)
        output = await tool.acall(**argument_dict)

        if verbose:
            print(f"Got output: {output!s}")
            print("========================")
    else:
        err_msg = build_missing_tool_message(name)
        output = build_error_tool_output(name, arguments_str, err_msg)

        if verbose:
            print(err_msg)
            print("========================")

    return (
        ChatMessage(
            content=str(output),
            role=MessageRole.FUNCTION,
            additional_kwargs={
                "name": name,
            },
        ),
        output,
    )


class GeminiAssistantAgent(BaseAgent):
    """Gemini Assistant agent.
    
    Wrapper around Google's Gemini API
    """

    def __init__(
        self,
        client: Any,
        model: str,
        tools: Optional[List[BaseTool]],
        callback_manager: Optional[CallbackManager] = None,
        chat_history: Optional[List[Any]] = None,
        instructions: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize Gemini assistant agent.
        
        Args:
            client: Gemini API client
            model: Gemini model to use (e.g., "gemini-pro")
            tools: List of tools to use with function calling
            callback_manager: Callback manager
            chat_history: Initial chat history
            instructions: System instructions for the model
            verbose: Whether to print verbose logs
        """
        self._client = client
        self._model = model
        self._tools = tools or []
        self._chat_history = chat_history or []
        self._instructions = instructions
        self._verbose = verbose
        
        # Create a Gemini model instance
        self._gemini_model = self._client.GenerativeModel(model_name=model)
        
        # Initialize chat session if there's history or instructions
        if self._chat_history or self._instructions:
            self._initialize_chat_session()
        else:
            self._chat_session = None
            
        self.callback_manager = callback_manager or CallbackManager([])

    def _initialize_chat_session(self) -> None:
        """Initialize a new chat session with Gemini."""
        # Create a new chat session
        chat = self._gemini_model.start_chat()
        
        # Add system instructions if provided
        if self._instructions:
            chat.send_message({"role": "user", "parts": [{"text": f"SYSTEM: {self._instructions}"}]})
            
        # Add existing chat history if provided
        for message in self._chat_history:
            if isinstance(message, dict):
                chat.send_message(message)
            else:
                # Assume it's a ChatMessage
                gemini_message = to_gemini_message(message)
                chat.send_message(gemini_message)
                
        self._chat_session = chat

    @classmethod
    def from_new(
        cls,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        model: str = "gemini-pro",
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> "GeminiAssistantAgent":
        """Create a new Gemini Assistant Agent.
        
        Args:
            instructions: System instructions for the model
            tools: List of tools
            model: Gemini model to use
            callback_manager: Callback manager
            verbose: Whether to print verbose logs
            api_key: Gemini API key
            temperature: Temperature parameter for generation
        """
        import google.generativeai as genai
        
        # Initialize client with API key
        if api_key:
            genai.configure(api_key=api_key)
        
        # Convert tools for Gemini if provided
        gemini_tools = []
        if tools:
            for tool in tools:
                function_def = tool.metadata.to_dict()
                gemini_tools.append({
                    "function_declarations": [{
                        "name": function_def.get("name"),
                        "description": function_def.get("description"),
                        "parameters": function_def.get("parameters", {})
                    }]
                })
        
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        
        return cls(
            client=genai,
            model=model,
            tools=tools,
            callback_manager=callback_manager,
            instructions=instructions,
            verbose=verbose,
        )

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history."""
        if self._chat_session:
            return from_gemini_messages(self._chat_session.history)
        return []

    def reset(self) -> None:
        """Reset chat history."""
        self._chat_history = []
        if self._instructions:
            self._initialize_chat_session()
        else:
            self._chat_session = None

    def get_tools(self, message: str) -> List[BaseTool]:
        """Get tools."""
        return self._tools

    def add_message(self, message: str) -> Any:
        """Add message to chat history."""
        # Initialize chat session if not already done
        if not self._chat_session:
            self._initialize_chat_session()
            
        # Send message to Gemini
        result = self._chat_session.send_message(message)
        return result

    def _process_function_calling(self, response: Any) -> List[ToolOutput]:
        """Process function calling from Gemini response."""
        tool_outputs = []
        
        # Check if response has function_call
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            function_call = {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args)
                            }
                            _, tool_output = call_function(
                                self._tools, function_call, verbose=self._verbose
                            )
                            tool_outputs.append(tool_output)
                            
                            # Add function response back to chat
                            function_response = {
                                "role": "function",
                                "parts": [{"text": str(tool_output)}],
                                "name": part.function_call.name
                            }
                            self._chat_session.send_message(function_response)
        
        return tool_outputs

    async def _aprocess_function_calling(self, response: Any) -> List[ToolOutput]:
        """Process function calling from Gemini response asynchronously."""
        tool_outputs = []
        
        # Check if response has function_call
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            function_call = {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args)
                            }
                            _, tool_output = await acall_function(
                                self._tools, function_call, verbose=self._verbose
                            )
                            tool_outputs.append(tool_output)
                            
                            # Add function response back to chat
                            function_response = {
                                "role": "function",
                                "parts": [{"text": str(tool_output)}],
                                "name": part.function_call.name
                            }
                            self._chat_session.send_message(function_response)
        
        return tool_outputs

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""
        # Send message to Gemini
        response = self.add_message(message)
        
        # Process function calls if present
        sources = self._process_function_calling(response)
        
        # Get response content
        response_text = ""
        if hasattr(response, "text"):
            response_text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            response_text += part.text
        
        return AgentChatResponse(
            response=response_text,
            sources=sources,
        )

    async def _achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Asynchronous main chat interface."""
        # Send message to Gemini
        response = self.add_message(message)
        
        # Process function calls if present
        sources = await self._aprocess_function_calling(response)
        
        # Get response content
        response_text = ""
        if hasattr(response, "text"):
            response_text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            response_text += part.text
        
        return AgentChatResponse(
            response=response_text,
            sources=sources,
        )

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message, chat_history, function_call, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = await self._achat(
                message, chat_history, function_call, mode=ChatResponseMode.WAIT
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        function_call: Union[str, dict] = "auto",
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")
