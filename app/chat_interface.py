"""
Chat interface for Streamlit app.

This module provides functions for handling chat interactions, streaming responses,
and message display.
"""
import streamlit as st
from juena.clients.client import AgentClient, AgentClientError
from juena.schema.server import ChatMessage
from app.ui_components import (
    render_header_with_logo,
    render_message,
    render_streaming_token,
)


def process_stream_chunk(
    chunk,
    client: AgentClient,
    response_text: str,
    received_complete_message: bool,
    message_placeholder,
    messages: list
) -> tuple[str, bool]:
    """
    Process a single chunk from the stream and update UI.
    
    Args:
        chunk: Stream chunk (ChatMessage or token dict)
        client: AgentClient instance for helper methods
        response_text: Accumulated response text
        received_complete_message: Whether complete message was received
        message_placeholder: Streamlit placeholder
        messages: Message history list
        
    Returns:
        Tuple of (updated_response_text, updated_complete_flag)
    """
    if isinstance(chunk, ChatMessage):
        # Complete message received
        content_str = str(chunk.content) if chunk.content is not None else ""
        # Skip stray 'Start' messages
        if content_str.strip().lower() == "start":
            return response_text, received_complete_message
        
        received_complete_message = True
        # Backend handles deduplication, so we can always append
        messages.append(chunk)
        
        if chunk.type == "ai":
            message_placeholder.markdown(chunk.content)
            response_text = chunk.content
        
        return response_text, received_complete_message
    
    elif client.is_token_message(chunk):
        # Token message - accumulate
        if not received_complete_message:
            token_content = client.get_token_content(chunk) or ""
            
            if token_content:
                response_text += token_content
                render_streaming_token(response_text, message_placeholder)
        
        return response_text, received_complete_message
    
    return response_text, received_complete_message


def render_chat_interface() -> None:
    """Render the main chat interface including header, message history, and input."""
    # Render header
    render_header_with_logo()
    
    # Auto-trigger initial welcome from server when connected and history is empty
    if (
        st.session_state.server_connected
        and st.session_state.client
        and not st.session_state.messages
        and not st.session_state.get("welcome_initialized", False)
    ):
        st.session_state.welcome_initialized = True
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""
            received_complete_message = False

            try:
                for chunk in st.session_state.client.stream(
                    message="Start",
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_id,
                    provider=st.session_state.selected_provider,
                    model=st.session_state.selected_model,
                    stream_tokens=True,
                ):
                    # Process chunk using unified helper
                    response_text, received_complete_message = process_stream_chunk(
                        chunk,
                        st.session_state.client,
                        response_text,
                        received_complete_message,
                        message_placeholder,
                        st.session_state.messages
                    )

            except AgentClientError as e:
                st.error(f"Error communicating with server: {e}")
                error_message = ChatMessage(type="ai", content=f"Error: {str(e)}")
                st.session_state.messages.append(error_message)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                error_message = ChatMessage(type="ai", content=f"Unexpected error: {str(e)}")
                st.session_state.messages.append(error_message)
            finally:
                # Finalize message if we streamed tokens only
                if response_text and not received_complete_message:
                    message_placeholder.markdown(response_text)
                    ai_message = ChatMessage(type="ai", content=response_text)
                    st.session_state.messages.append(ai_message)
                st.rerun()

    # Display chat history (filter out system messages unless debug mode is enabled)
    for message in st.session_state.messages:
        render_message(message, show_system=st.session_state.show_system_messages)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.client:
            st.error("Server not connected. Please check server status in the sidebar.")
        else:
            # Add user message
            user_message = ChatMessage(type="human", content=prompt)
            st.session_state.messages.append(user_message)
            render_message(user_message)

            # Stream response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response_text = ""
                received_complete_message = False

                try:
                    for chunk in st.session_state.client.stream(
                        message=prompt,
                        thread_id=st.session_state.thread_id,
                        user_id=st.session_state.user_id,
                        provider=st.session_state.selected_provider,
                        model=st.session_state.selected_model,
                        stream_tokens=True
                    ):
                        # Process chunk using unified helper
                        response_text, received_complete_message = process_stream_chunk(
                            chunk,
                            st.session_state.client,
                            response_text,
                            received_complete_message,
                            message_placeholder,
                            st.session_state.messages
                        )

                    # Finalize message display
                    # Only add accumulated token text if we didn't receive a complete ChatMessage
                    if response_text and not received_complete_message:
                        message_placeholder.markdown(response_text)
                        ai_message = ChatMessage(type="ai", content=response_text)
                        st.session_state.messages.append(ai_message)
                    elif not response_text and st.session_state.messages:
                        # If no response text accumulated, check for last message
                        last_msg = st.session_state.messages[-1]
                        if isinstance(last_msg, ChatMessage) and last_msg.type == "ai":
                            message_placeholder.markdown(last_msg.content)
                    
                    # After streaming completes, rerun to display any tool messages that were added
                    # This ensures all messages (including tool messages) are displayed in correct order
                    st.rerun()

                except AgentClientError as e:
                    st.error(f"Error communicating with server: {e}")
                    error_message = ChatMessage(
                        type="ai",
                        content=f"Error: {str(e)}"
                    )
                    st.session_state.messages.append(error_message)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    error_message = ChatMessage(
                        type="ai",
                        content=f"Unexpected error: {str(e)}"
                    )
                    st.session_state.messages.append(error_message)
