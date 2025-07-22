import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";

export function getGroqLLM() {
  return new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama3-70b-8192",
  });
}

export function buildChatPrompt(context) {
  return ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant. Use the following context to answer the user's question as accurately as possible. Dont mention the context in your response. just answer the question."],
    ["human", "Context:\n" + context + "\n\nUser question: {input}"]
  ]);
} 