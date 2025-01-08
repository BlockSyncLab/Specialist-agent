// trainAI.ts
import fs from 'fs';
import path from 'path';
import { ChatOpenAI } from "@langchain/openai";

// Função para treinar a IA com informações de um arquivo TXT
export async function trainAI(fileName: string) {
  try {
    // Obtém o diretório atual usando import.meta.url
    const __filename = new URL(import.meta.url).pathname;
    const __dirname = path.dirname(__filename);  // Deriva o diretório

    // Caminho para o arquivo TXT
    const filePath = path.join(__dirname, 'info', fileName);

    // Verifica se o arquivo existe
    if (!fs.existsSync(filePath)) {
      throw new Error(`Arquivo não encontrado: ${filePath}`);
    }

    // Lê o conteúdo do arquivo
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    console.log("Conteúdo do arquivo carregado:", fileContent);

    // Inicializa o modelo com a chave da API do OpenAI
    const model = new ChatOpenAI({
      model: "gpt-4o-mini", // ou outro modelo disponível
      temperature: 0,
      openAiApiKey: process.env.OPENAI_API_KEY, // Certifique-se de ter configurado a chave API
    });

    // Cria uma mensagem com o conteúdo do arquivo para "treinar" o modelo
    const response = await model.invoke([fileContent]);

    console.log("IA treinada com sucesso, resposta:", response);
    return response;
  } catch (error) {
    console.error("Erro ao treinar a IA:", error);
    throw new Error("Falha ao treinar a IA");
  }
}
