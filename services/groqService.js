import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";

export function getGroqLLM() {
  return new ChatGroq({
    apiKey: process.env.GROQ_API_KEY,
    model: "llama3-70b-8192",
  });
}

export async function getAnswerFromContext(context, question) {
  const llm = getGroqLLM();
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant. Use the following context to answer the user's question as accurately as possible. Dont mention the context in your response. just answer the question."],
    ["human", "Context:\n" + context + "\n\nUser question: {input}"]
  ]);
  const chain = prompt.pipe(llm);
  const response = await chain.invoke({ input: question });
  return response.content;
}

export async function summarizeAndRephrase(text) {
  const llm = getGroqLLM();
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant. Elaborate the answer to be more clear and concise. Give detailed answer. Just give out content , dont add description or anything else"],
    ["human", "Answer: {input}"]
  ]);
  const chain = prompt.pipe(llm);
  const response = await chain.invoke({ input: text });
  return response.content;
} 