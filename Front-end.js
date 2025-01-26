import React, { useState } from "react";
import { motion } from "framer-motion";

const ChatbotApp = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const sendMessage = async () => {
    if (!userInput.trim()) return;

    const newMessage = { role: "user", content: userInput };
    setMessages([...messages, newMessage]);
    setUserInput("");

    // Simulate bot response (replace this with API call to your chatbot)
    const botResponse = { role: "assistant", content: "That's interesting. Can you tell me more?" };
    setTimeout(() => setMessages((prev) => [...prev, botResponse]), 1000);
  };

  return (
    <div className="bg-gradient-to-b from-blue-100 via-green-100 to-gray-100 min-h-screen flex flex-col justify-between items-center p-4">
      <header className="text-2xl font-semibold text-gray-800 mb-4"> Hey Im (whatever)! Your personal mental Health assistant! :)</header>
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg p-4 flex-grow flex flex-col overflow-hidden">
        <div className="flex-grow overflow-y-auto mb-4">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className={`mb-2 p-2 rounded-lg max-w-xs ${
                msg.role === "user" ? "ml-auto bg-blue-200" : "mr-auto bg-green-200"
              }`}
            >
              {msg.content}
            </motion.div>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Type your message..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            className="flex-grow p-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-300"
          />
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={sendMessage}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg shadow-md hover:bg-blue-600 focus:outline-none focus:ring focus:ring-blue-300"
          >
            Send
          </motion.button>
        </div>
      </div>
      <footer className="text-sm text-gray-600 mt-4">© 2025 Well-being Chatbot. All rights reserved.</footer>
    </div>
  );
};

export default ChatbotApp;
