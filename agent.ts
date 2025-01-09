import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import express, { Request, Response } from "express";
import { DateTime } from "luxon";
import fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';
import { trainAI } from './trainAI';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const openaiApiKey = process.env.OPENAI_API_KEY;
const tavilyApiKey = process.env.TAVILY_API_KEY;

const app = express();

app.use(express.json());

const tools = [new TavilySearchResults({ maxResults: 3 })];
const toolNode = new ToolNode(tools);

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools);

function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1];
  if (lastMessage.additional_kwargs.tool_calls) {
    return "tools";
  }
  return "__end__";
}

async function callModel(state: typeof MessagesAnnotation.State) {
  console.log("Model Input State:", state);
  const response = await model.invoke(state.messages);
  console.log("Model Response:", response);
  return { messages: [response] };
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

async function loadDataCenterFiles() {
  const infoDir = path.join(__dirname, 'info');
  const files = fs.readdirSync(infoDir);
  const fileContents = [];
  for (const file of files) {
    const filePath = path.join(infoDir, file);
    if (file.endsWith('.txt')) {
      const textContent = fs.readFileSync(filePath, 'utf-8');
      fileContents.push(textContent);
    }
  }
  return fileContents.join('\n');
}

function loadPrompt(agent: string, question: string, date: string, datacenterContent: string) {
  const agentFilePath = path.join(__dirname, "Crew", `${agent}.txt`);
  if (!fs.existsSync(agentFilePath)) {
    throw new Error(`Agent configuration file not found: ${agentFilePath}`);
  }
  const agentPromptTemplate = fs.readFileSync(agentFilePath, "utf-8");
  return agentPromptTemplate
    .replace(/{{DATE}}/g, date)
    .replace(/{{QUESTION}}/g, question)
    .replace(/{{DATACENTER}}/g, datacenterContent);
}

app.post("/ask", async (req: Request, res: Response) => {
  const { question, agent } = req.body;
  if (!question || !agent) {
    return res.status(400).send({ error: "Question and Agent are required" });
  }
  try {
    const currentDate = DateTime.now().toFormat("dd/MM/yyyy");
    const datacenterContent = await loadDataCenterFiles();
    const fileName = `datacenter.txt`;
    await trainAI(fileName);
    const prompt = loadPrompt(agent, question, currentDate, datacenterContent);
    console.log("Final Prompt Sent to Model:", prompt);
    const finalState = await workflow.compile().invoke({
      messages: [new HumanMessage(prompt)],
    });
    console.log("Final State Messages:", JSON.stringify(finalState.messages, null, 2));
    const responseMessage = finalState.messages[finalState.messages.length - 1];
    const summary =
      responseMessage.raw_response?.summary || responseMessage.content || "No response content found";
    console.log("Response Summary:", summary);
    res.json({
      response: summary,
    });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send({ error: "Internal Server Error" });
  }
});

const port = process.env.PORT || 8080;

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
