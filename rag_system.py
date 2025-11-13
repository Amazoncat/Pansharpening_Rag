from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import pickle
import numpy as np
from openai import OpenAI

class RagSystem:
    def __init__(self,api_key: str,docs_dir: str ='docs'):
        self.api_key=api_key
        self.docs_dir=Path(docs_dir)
        self.client=OpenAI(
            base_url='https://api.siliconflow.cn/v1',
            api_key=api_key
        )
        self.documents=[]        
        self.document_chunks=[] 
        self.document_vectors=None  
        self.vectorizer=None     

        self.stop_words=self.load_stop_words()      #停用词列表

        self.cache_dir=Path('cache')                    #缓存目录
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.chunk_cache=self.cache_dir/'chunk_cache.pkl' #文档块缓存
        self.vector_cache=self.cache_dir/'vector_cache.pkl' #向量缓存

    def initialize(self):
        print("正在进行初始化...")
        self.load_docs()
        self.prepare_documents()
        self.build_vecotr_index()
        print("初始化已完成") 

    def load_stop_words(self):
        '''加载停用词'''
        stop_words_file=Path('stop_words.txt')
        stop_words=set()
        if not stop_words_file.exists():
            print(f"停用词文件路径不存在: {stop_words_file}")
            return []
        try:
            with open(stop_words_file,'r',encoding='utf-8') as f:
                for line in f:
                    word=line.strip()
                    if word:
                        stop_words.add(word)
        except Exception as e:
            print(f"Error loading stop words:错误为 {e}")
            return []
        print(f"成功加载 {len(stop_words)} 个停用词")
        return stop_words

    def chinese_tokenizer(self,text:str):
        '''中文分词器'''
        words=list(jieba.cut(text))
        # print(words)
        punctuation='，。！？：；“”‘’（）【】《》、'
        filtered_words=[]
        for word in words:
            word=word.strip()
            if word and word not in self.stop_words and len(word)>1 and word not in punctuation:
                filtered_words.append(word)
        return filtered_words
        
    def build_vecotr_index(self):
        # 如果向量缓存文件存在，则从缓存文件中加载向量索引
        if self.vector_cache.exists():
            print(f"发现向量缓存文件，从缓存文件中加载向量索引...")
            with open(self.vector_cache,'rb') as f:
                cache_data=pickle.load(f)
                self.vectorizer=TfidfVectorizer(
                    tokenizer=self.chinese_tokenizer,
                    vocabulary=cache_data['vocabulary'],
                )
                self.vectorizer.idf_=cache_data['idf']
                self.document_vectors=cache_data['vectors']
            print(f"成功加载向量索引")
            return
        
            # 如果向量缓存文件不存在，则构建向量索引
        print("未发现向量缓存文件，正在构建向量索引...")    
        self.vectorizer=TfidfVectorizer(
            tokenizer=self.chinese_tokenizer,
            max_features=5000,#最大特征数
            min_df=2,#最小文档频率 忽略出现次数小于2的词语
            max_df=0.8,#最大文档频率 忽略出现次数大于80%的词语
            stop_words=None,#停用词 忽略停用词
            use_idf=True,#是否使用idf 使用idf权重
            smooth_idf=True,#是否使用平滑idf 使用平滑idf权重
            sublinear_tf=False,#是否使用线性tf 使用线性tf权重
            norm='l2',#归一化 使用l2归一化
            lowercase=True,#是否转换为小写 转换为小写
            encoding='utf-8',#编码 使用utf-8编码
            ngram_range=(1,3)#ngram范围 使用1-3元语法
            )
        # 构建文档向量  
        text_data=[chunk['content'] for chunk in self.document_chunks]
        self.document_vectors=self.vectorizer.fit_transform(text_data)

        cache_data={'vectors':self.document_vectors,
                    'vocabulary':self.vectorizer.vocabulary_,
                    'idf':self.vectorizer.idf_}

        with open(self.vector_cache,'wb') as f:
            pickle.dump(cache_data,f)
        print(f"成功构建向量索引并保存到缓存文件中")

    def load_docs(self):
        # 加载文档
        print("正在加载论文...")
        for file_path in self.docs_dir.glob("*.txt"):
            # print(file_path)
            try:
                with open (file_path,'r',encoding='utf-8') as f :
                    content=f.read().strip()
                    # print(content)
                    if content:
                        self.documents.append({
                            'filename':file_path.name,
                            'content':content,
                            'path':str(file_path)
                        })
            except Exception as e:
                print(f"Error loading {file_path}:错误为 {e}")
        print(f"成功加载 {len(self.documents)} 篇论文")
    
    def prepare_documents(self):
        if self.chunk_cache.exists():
            print(f"发现缓存文件，从缓存文件中加载文档块...")
            with open(self.chunk_cache,'rb') as f:
                self.document_chunks=pickle.load(f)
            print(f"成功加载 {len(self.document_chunks)} 个文档块")
            return
        
        print("没有发现缓存文件，正在分割文档块...")
        for doc in self.documents:
            chunks=self.split_documents_chunks(doc['content'])
            for i,chunk in enumerate(chunks):
                self.document_chunks.append({
                    'source':doc['filename'],
                    'chunk_index':i,
                    'content':chunk,
                    'full_path':doc['path']
                })

        

        #保存缓存文件
        with open(self.chunk_cache,'wb') as f:
            pickle.dump(self.document_chunks,f)


        print(f"成功分割 {len(self.document_chunks)} 个文档块")
        # print(self.document_chunks)
    
    def split_documents_chunks(self,text:str,chunk_size:int=300):
        """
        将文档分割成块
        """
        sentences=re.split(r'[.!?。！？]',text)
        sentences=[sentence.strip() for sentence in sentences if sentence.strip()]
        current_chunk=""
        chunks=[]
        for sentence in sentences:
            if len(current_chunk)+len(sentence)<=chunk_size: #根据句子长度和当前块长度判断是否可以添加到当前块
                current_chunk+=sentence+"."
            else:
                if current_chunk: #如果当前块不为空（存在），则添加到chunks，拼接字符串够了
                    chunks.append(current_chunk.strip())
                current_chunk=sentence+"."
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def retrieve(self, query: str, top_k: int = 10,similarity_threshold: float = 0.3):
        """
        检索与查询最相关的文档块
        """
        if self.vectorizer is None or self.document_vectors is None:
            print("向量索引未初始化，请先调用initialize()/build_vecotr_index方法")
            return []
        
        # 将查询转换为向量          
        query_vector = self.vectorizer.transform([query])
        
        # 计算查询向量与所有文档向量的相似度（余弦相似度）[0,1]
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        # 过滤相似度小于相似度阈值的文档块
        top_indices = np.where(similarities >= similarity_threshold)[0]

        # 按相似度排序从高到低
        sorted_indices = np.argsort(similarities)[::-1] 
        # 获取top_k个最相似的文档块索引
        top_indices = sorted_indices[:top_k]
        
        results=[]
        for idx in top_indices:
            chunk=self.document_chunks[idx].copy()
            chunk['similarity']=float(similarities[idx])
            results.append(chunk)
        return results
    
    def query(self, question: str, top_k: int = 10,similarity_threshold: float = 0.01):
        """
        查询并返回相关文档块
        """
        results = self.retrieve(question, top_k)
        if not results:
            print("未找到相关文档。\n")
            return {
                'question':question,
                'answer':'抱歉，未找到相关文档。',
                'sources':[]    
            }
        print(f"\n找到 {len(results)} 个相关文档块：\n")

        print(f"正在生成答案...")
        print(f"答案: {self.generate_answer(question,results)}") 
        answer=self.generate_answer(question,results)

        #生成来源信息
        sources=[]
        for chunk in results:
            sources.append({
                'source':chunk['source'],
                'similarity':chunk['similarity'],
                'content_preview':chunk['content'][:100]+'...' if len(chunk['content'])>100 else chunk['content']
            })
        return {
            'question':question,
            'answer':answer,
            'sources':sources
        }

    def generate_answer(self,question:str,context_chunks:list):
        """
        生成答案
        """
        context="\n\n".join([f" 文档片段{i+1}: {chunk['content']}" for i,chunk in enumerate(context_chunks)])


        system_prompt="""
        # Role: 红楼梦研究专家

        ## Profile
        - language: 中文
        - description: 精通《红楼梦》文本及红学研究，能够深入解析作品的人物、情节、诗词及文化内涵
        - background: 多年从事《红楼梦》研究与教学，参与过多项红学课题研究
        - personality: 严谨细致，富有文人气质，善于引经据典
        - expertise: 文本分析、人物研究、诗词鉴赏、文化解读
        - target_audience: 红学爱好者、文学研究者、学生群体

        ## Skills

        1. 文本解析能力
        - 情节梳理: 能准确还原小说情节脉络
        - 细节把握: 对文本细节有敏锐洞察力
        - 人物分析: 深入剖析人物性格与命运
        - 诗词解读: 精准解析书中诗词内涵

        2. 学术研究能力
        - 文献考证: 熟悉各类红学研究成果
        - 文化阐释: 揭示作品背后的文化内涵
        - 比较研究: 能与其他文学作品进行对比
        - 版本鉴别: 了解不同版本差异

        ## Rules

        1. 回答原则：
        - 基于文本: 所有回答必须严格依据原著文本
        - 严谨准确: 不妄加猜测，不传播未经考证的观点
        - 深度解析: 透过表面现象揭示深层含义
        - 客观公正: 避免个人主观臆断

        2. 行为准则：
        - 引经据典: 重要观点需引用原文佐证
        - 语言典雅: 保持与原著相称的文雅风格
        - 层次分明: 回答要有逻辑性和条理性
        - 深入浅出: 复杂问题要解释得通俗易懂

        3. 限制条件：
        - 不涉争议: 避免介入红学争议性话题
        - 不妄评续作: 对后四十回保持审慎态度
        - 不越文本: 不脱离文本过度解读
        - 不代作者: 不以作者口吻发表观点

        ## Workflows

        - 目标: 提供专业准确的红楼梦解读
        - 步骤 1: 仔细理解用户问题，明确询问重点
        - 步骤 2: 检索相关文本段落，确认信息准确性
        - 步骤 3: 组织回答内容，适当引用原文
        - 步骤 4: 以典雅文风呈现完整解答
        - 预期结果: 用户获得权威、深入的红楼梦知识

        ## Initialization
        作为红楼梦研究专家，你必须遵守上述Rules，按照Workflows执行任务。
        """.strip()
        user_prompt=f"""基于以下文档片段，回答用户问题：
        用户问题: {question}
        文档片段: {context}
        """.strip()
        try:
            response=self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V2.5",
                messages=[
                    {"role":"system",
                    "content":system_prompt},
                    {"role":"user",
                    "content":user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                    stream=False
                )
            return response.choices[0].message.content
        except Exception as e:
            error_type=type(e).__name__
            error_detail=str(e) if str(e) else "未知错误"
            print(f"生成答案失败: {error_type}: {error_detail}")
            return f"生成答案失败: {error_type}: {error_detail}"
        
def main():
    rag_system=RagSystem("sk-hafoqeioagkxevnggctitistlcfsiioxdqktzdjtpdlawuoz")  #deepseek api key
    rag_system.initialize()



    #交互式问答
    print("\n" + "="*50)
    print("RAG系统已就绪，开始交互式问答")
    print("输入 'quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    while True:
        if True:
            question = input("请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用，再见！")
                break
            
            # 检索相关文档块
            results = rag_system.query(question, top_k=5)
            

            # print(f"问题: {results['question']}")
            print(f"\n答案: {results['answer']}")
            if results['sources']:
                print(f"\n相关文档块:")
                for i,source in enumerate(results['sources'],1):
                    print(f"[结果 {i}] (相似度: {source['similarity']:.4f})")
                    print(f"来源: {source['source']}")
                    print(f"内容预览: {source['content_preview']}")
                    print("-" * 50)
                print()
        #     if not results:
        #         print("未找到相关文档。\n")
        #         continue
            
        #     print(f"\n找到 {len(results)} 个相关文档块：\n")
        #     for i, result in enumerate(results, 1):
        #         chunk = result['chunk']
        #         similarity = result['similarity']
        #         print(f"[结果 {i}] (相似度: {similarity:.4f})")
        #         print(f"来源: {chunk['source']}")
        #         print(f"内容: {chunk['content'][:200]}...")  # 只显示前200个字符
        #         print("-" * 50)
            
        #     print()  # 空行
            
        # except KeyboardInterrupt:
        #     print("\n\n感谢使用，再见！")
        #     break
        # except Exception as e:
        #     print(f"发生错误: {e}\n")

if __name__ == '__main__':
    main()
