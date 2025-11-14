import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import './ChatbotInterface.css';
import BotAvatar from './assets/ChatBot_Pic.png';
import UserAvatar from './assets/Customer_Pic.png';

export default function ChatbotInterface() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! Welcome to our customer service. I'm here to assist you with your order inquiry today. How can I help you?",
      sender: 'bot',
      timestamp: '10:00 AM'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleReset = () => {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
    setMessages([
      {
        id: 1,
        text: "Hello! Welcome to our customer service. I'm here to assist you with your order inquiry today. How can I help you?",
        sender: 'bot',
        timestamp: timeString
      }
    ]);
    setInputValue('');
  };

  const handleSend = () => {
    if (inputValue.trim() === '') return;

    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });

    const newMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: 'user',
      timestamp: timeString
    };

    setMessages([...messages, newMessage]);
    setInputValue('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <div className="header-content">
          <div className="avatar-wrapper">
            <div className="avatar-outer">
              <img src={BotAvatar} alt="Bot Avatar" className="avatar-image" />
            </div>
          </div>
          <div>
            <h1 className="header-title">Customer Service Assistant</h1>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span className="status-text">Online - Ready to help</span>
            </div>
          </div>
          <button className="reset-button" onClick={handleReset}>
            Reset Chat
          </button>
        </div>
      </div>

      {/* Messages Container */}
      <div className="messages-container">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message-row ${message.sender === 'user' ? 'message-row-user' : 'message-row-bot'}`}
          >
            <div className={`message-wrapper ${message.sender === 'user' ? 'message-wrapper-user' : 'message-wrapper-bot'}`}>
              {message.sender === 'bot' && (
                <div className="message-avatar bot-avatar">
                  <img src={BotAvatar} alt="Bot" className="avatar-image" />
                </div>
              )}
              {message.sender === 'user' && (
                <div className="message-avatar user-avatar">
                  <img src={UserAvatar} alt="User" className="avatar-image" />
                </div>
              )}
              <div>
                <div className={`message-bubble ${message.sender === 'user' ? 'message-bubble-user' : 'message-bubble-bot'}`}>
                  <p className="message-text">{message.text}</p>
                </div>
                <div className={`message-timestamp ${message.sender === 'user' ? 'timestamp-right' : 'timestamp-left'}`}>
                  {message.timestamp}
                </div>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="message-input"
          />
          <button onClick={handleSend} className="send-button">
            Send
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}