from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field


class ContentHashID(BaseModel):
    md5: str = Field(..., pattern=r"^[a-f0-9]{32}$")


class ContextHashID(BaseModel):
    md5: str = Field(..., pattern=r"^[a-f0-9]{32}$")


class RoleName(BaseModel):
    value: str


class ModelName(BaseModel):
    value: str = "anthropic/claude-sonnet-4.5"


class FunctionCallRequest(BaseModel):
    name: str
    arguments_json: str


class FunctionCallResult(BaseModel):
    name: str
    result_json: str


class EnabledFunction(BaseModel):
    name: str
    schema_hash: str = Field(..., pattern=r"^[a-f0-9]{32}$")


class PromptMessage(BaseModel):
    kind: Literal["prompt"] = "prompt"
    context_id: ContextHashID
    content_id: ContentHashID
    role: RoleName
    original_prompt: str
    augmented_prompt: str
    included_context: Sequence[ContextHashID] = Field(default_factory=list)
    enabled_functions: Sequence[EnabledFunction] = Field(default_factory=list)


class ResponseMessage(BaseModel):
    kind: Literal["response"] = "response"
    context_id: ContextHashID
    content_id: ContentHashID
    model: ModelName
    in_response_to: ContextHashID
    text: str
    function_calls: Sequence[FunctionCallRequest] = Field(default_factory=list)
    function_results: Sequence[FunctionCallResult] = Field(default_factory=list)
    thinking_text: str = ""


AnyMessage = PromptMessage | ResponseMessage


class SessionName(BaseModel):
    value: str


class SessionTurn(BaseModel):
    index: int
    prompt_id: ContextHashID
    response_id: Optional[ContextHashID] = None


class SessionFile(BaseModel):
    session: SessionName
    created_at_iso: str
    turns: Sequence[SessionTurn] = Field(default_factory=list)
    message_ids: Sequence[ContextHashID] = Field(default_factory=list)


class PluginFunctionMeta(BaseModel):
    kind: Literal["function_provider"] = "function_provider"
    name: str
    description: str = ""


class PluginMCPMeta(BaseModel):
    kind: Literal["mcp_client"] = "mcp_client"
    name: str
    description: str = ""


class PluginRagMeta(BaseModel):
    kind: Literal["rag_provider"] = "rag_provider"
    name: str
    description: str = ""


class PluginAgentMeta(BaseModel):
    kind: Literal["agent"] = "agent"
    name: str
    description: str = ""


PluginMetadata = PluginFunctionMeta | PluginMCPMeta | PluginRagMeta | PluginAgentMeta
