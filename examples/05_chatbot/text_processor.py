"""
文本預處理工具
用於處理和準備文本資料
"""

import re
from typing import List, Dict, Tuple
import jieba  # 中文分詞


class TextProcessor:
    """文本處理器"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        初始化文本處理器
        
        Args:
            chunk_size: 文本塊大小
            overlap: 重疊大小
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理後的文本
        """
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、數字和基本標點）
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）\[\]【】《》、]', '', text)
        
        # 移除連續的標點符號
        text = re.sub(r'[。，！？；：]{2,}', '。', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        將長文本切分為固定大小的chunks
        
        Args:
            text: 輸入文本
            
        Returns:
            文本塊列表
        """
        chunks = []
        text = self.clean_text(text)
        
        # 按句子切分，避免在句子中間切斷
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # 添加最後一個chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 創建有重疊的chunks
        if self.overlap > 0:
            chunks = self._create_overlapping_chunks(chunks)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        將文本切分為句子
        
        Args:
            text: 輸入文本
            
        Returns:
            句子列表
        """
        # 中文句子切分
        chinese_sentences = re.split(r'[。！？]', text)
        
        # 英文句子切分（如果有）
        sentences = []
        for sent in chinese_sentences:
            if sent.strip():
                # 進一步切分英文句子
                eng_sentences = re.split(r'[.!?]', sent)
                sentences.extend([s for s in eng_sentences if s.strip()])
        
        return sentences
    
    def _create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        創建有重疊的文本塊
        
        Args:
            chunks: 原始文本塊
            
        Returns:
            有重疊的文本塊
        """
        overlapping_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                overlapping_chunks.append(chunks[i])
            else:
                # 從前一個chunk取部分內容作為重疊
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
                overlapping_chunks.append(overlap_text + " " + chunks[i])
        
        return overlapping_chunks
    
    def load_documents(self, file_paths: List[str]) -> List[Dict]:
        """
        載入並處理多個文檔
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            處理後的文檔資料
        """
        all_documents = []
        
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # 切分文本
                chunks = self.split_into_chunks(text)
                
                # 為每個chunk創建文檔物件
                for i, chunk in enumerate(chunks):
                    doc = {
                        "content": chunk,
                        "metadata": {
                            "source": path,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    }
                    all_documents.append(doc)
                    
                print(f"已處理：{path} - {len(chunks)} 個文本塊")
                
            except Exception as e:
                print(f"處理文件 {path} 時出錯：{str(e)}")
        
        return all_documents
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取關鍵詞（使用簡單的詞頻方法）
        
        Args:
            text: 輸入文本
            top_k: 返回前k個關鍵詞
            
        Returns:
            關鍵詞列表
        """
        # 清理文本
        text = self.clean_text(text)
        
        # 中文分詞
        words = jieba.lcut(text)
        
        # 過濾停用詞（這裡只是簡單示例）
        stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '和', '與', '或'}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # 計算詞頻
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 排序並返回top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_k]]
    
    def calculate_text_statistics(self, text: str) -> Dict:
        """
        計算文本統計資訊
        
        Args:
            text: 輸入文本
            
        Returns:
            統計資訊字典
        """
        clean_text = self.clean_text(text)
        
        # 基本統計
        stats = {
            "total_chars": len(text),
            "clean_chars": len(clean_text),
            "sentences": len(self._split_sentences(clean_text)),
            "chinese_chars": len(re.findall(r'[\u4e00-\u9fff]', clean_text)),
            "english_chars": len(re.findall(r'[a-zA-Z]', clean_text)),
            "digits": len(re.findall(r'\d', clean_text)),
            "punctuation": len(re.findall(r'[。，！？；：""''（）\[\]【】《》、]', clean_text))
        }
        
        # 計算中英文比例
        total_content = stats["chinese_chars"] + stats["english_chars"]
        if total_content > 0:
            stats["chinese_ratio"] = stats["chinese_chars"] / total_content
            stats["english_ratio"] = stats["english_chars"] / total_content
        else:
            stats["chinese_ratio"] = 0
            stats["english_ratio"] = 0
        
        return stats


class DocumentProcessor(TextProcessor):
    """文檔處理器（擴展文本處理器）"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        super().__init__(chunk_size, overlap)
        self.supported_formats = ['.txt', '.md', '.json']
    
    def process_markdown(self, content: str) -> List[Dict]:
        """
        處理 Markdown 文檔
        
        Args:
            content: Markdown 內容
            
        Returns:
            處理後的文檔塊
        """
        documents = []
        
        # 按標題分割
        sections = re.split(r'\n#{1,6}\s', content)
        headers = re.findall(r'\n(#{1,6}\s[^\n]+)', content)
        
        for i, section in enumerate(sections):
            if section.strip():
                # 獲取對應的標題
                header = headers[i-1] if i > 0 and i-1 < len(headers) else ""
                
                # 切分段落
                chunks = self.split_into_chunks(section)
                
                for j, chunk in enumerate(chunks):
                    doc = {
                        "content": chunk,
                        "metadata": {
                            "type": "markdown",
                            "header": header.strip(),
                            "section_id": i,
                            "chunk_id": j
                        }
                    }
                    documents.append(doc)
        
        return documents
    
    def process_code(self, content: str, language: str = "python") -> List[Dict]:
        """
        處理程式碼文檔
        
        Args:
            content: 程式碼內容
            language: 程式語言
            
        Returns:
            處理後的程式碼塊
        """
        documents = []
        
        # 提取函數和類
        if language == "python":
            # 簡單的Python函數和類提取
            functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):[^}]+', content, re.MULTILINE)
            classes = re.findall(r'class\s+(\w+)[^:]*:[^}]+', content, re.MULTILINE)
            
            # 為每個函數或類創建文檔
            for func in functions:
                doc = {
                    "content": func,
                    "metadata": {
                        "type": "code",
                        "language": language,
                        "element_type": "function",
                        "name": func
                    }
                }
                documents.append(doc)
            
            for cls in classes:
                doc = {
                    "content": cls,
                    "metadata": {
                        "type": "code",
                        "language": language,
                        "element_type": "class",
                        "name": cls
                    }
                }
                documents.append(doc)
        
        return documents


def main():
    """主函數 - 使用範例"""
    
    print("=== 文本處理器範例 ===\n")
    
    # 創建文本處理器
    processor = TextProcessor(chunk_size=100, overlap=20)
    
    # 範例文本
    sample_text = """
    人工智慧（Artificial Intelligence，簡稱AI）是電腦科學的一個分支。
    它企圖了解智慧的實質，並生產出一種新的能以人類智慧相似的方式做出反應的智慧機器。
    該領域的研究包括機器人、語言識別、圖像識別、自然語言處理和專家系統等。
    人工智慧從誕生以來，理論和技術日益成熟，應用領域也不斷擴大。
    """
    
    # 清理文本
    clean_text = processor.clean_text(sample_text)
    print(f"清理後的文本：\n{clean_text}\n")
    
    # 切分文本
    chunks = processor.split_into_chunks(sample_text)
    print(f"文本切分結果（{len(chunks)} 個塊）：")
    for i, chunk in enumerate(chunks):
        print(f"  塊 {i+1}: {chunk[:50]}...")
    
    # 提取關鍵詞
    keywords = processor.extract_keywords(sample_text, top_k=5)
    print(f"\n關鍵詞：{keywords}")
    
    # 文本統計
    stats = processor.calculate_text_statistics(sample_text)
    print(f"\n文本統計：")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== 文檔處理器範例 ===\n")
    
    # 創建文檔處理器
    doc_processor = DocumentProcessor()
    
    # 處理 Markdown
    markdown_content = """
    # 標題一
    這是第一段內容。
    
    ## 子標題
    這是子標題下的內容。
    
    # 標題二
    這是第二個主要部分。
    """
    
    md_docs = doc_processor.process_markdown(markdown_content)
    print(f"Markdown 處理結果（{len(md_docs)} 個文檔）")
    for doc in md_docs[:2]:  # 只顯示前兩個
        print(f"  內容: {doc['content'][:30]}...")
        print(f"  元資料: {doc['metadata']}")


if __name__ == "__main__":
    # 需要安裝 jieba: pip install jieba
    main()