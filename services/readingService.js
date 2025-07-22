import fs from 'fs';
import path from 'path';
import pdf from 'pdf-parse';

export async function loadPDF(pdfRelativePath = 'public/manual.pdf') {
  const pdfPath = path.join(process.cwd(), pdfRelativePath);
  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdf(dataBuffer);
  console.log(`[PDF] Loaded and parsed ${pdfPath}`);
  return data.text;
} 