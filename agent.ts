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
import { trainAI } from './trainAI';  // Importando a função de treinamento

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);  // Corrige o problema com __dirname em módulos ES
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

// Função para carregar arquivos da pasta 'info' (somente arquivos TXT)
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

  return fileContents.join('\n');  // Unir o conteúdo de todos os arquivos
}

// Load prompt dinamicamente para o agente especificado
function loadPrompt(agent: string, question: string, date: string, datacenterContent: string) {
  const agentFilePath = path.join(__dirname, "Crew", `${agent}.txt`);
  
  console.log("Loading Prompt from:", agentFilePath);

  if (!fs.existsSync(agentFilePath)) {
    throw new Error(`Agent configuration file not found: ${agentFilePath}`);
  }

  const agentPromptTemplate = fs.readFileSync(agentFilePath, "utf-8");
  console.log("Loaded Prompt Template:", agentPromptTemplate);

  // Adiciona o conteúdo do datacenter ao prompt
  return agentPromptTemplate
    .replace(/{{DATE}}/g, date)
    .replace(/{{QUESTION}}/g, question)
    .replace(/{{DATACENTER}}/g, datacenterContent);  // Substituindo {{DATACENTER}} pelo conteúdo do treinamento
}

app.post("/ask", async (req: Request, res: Response) => {
  const { question, agent } = req.body;

  if (!question || !agent) {
    console.log("Validation Error: Missing question or agent");
    return res.status(400).send({ error: "Question and Agent are required" });
  }

  try {
    const currentDate = DateTime.now().toFormat("dd/MM/yyyy");

    console.log("Question:", question);
    console.log("Agent:", agent);
    console.log("Date:", currentDate);

    // Carregar o conteúdo de todos os arquivos da pasta 'info'
    const datacenterContent = await loadDataCenterFiles();

    // Treinar a IA com o conteúdo de datacenter.txt (ou outros arquivos)
    const fileName = `datacenter.txt`;  // Nome fixo do arquivo de treinamento
    await trainAI(fileName);  // Chama a função de treinamento

    // Gerar o prompt com o conteúdo do datacenter e a pergunta
    const prompt = loadPrompt(agent, question, currentDate, datacenterContent);
    console.log("Final Prompt Sent to Model:", prompt);

    const finalState = await workflow.compile().invoke({
      messages: [new HumanMessage(prompt)],
    });

    console.log(
      "Response:",
      finalState.messages[finalState.messages.length - 1].content
    );

    res.json({
      response:
        finalState.messages[finalState.messages.length - 1].content,
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
