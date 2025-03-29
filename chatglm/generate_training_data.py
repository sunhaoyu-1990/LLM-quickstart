import os
import json
import csv
import time
from typing import List, Dict
import re
import openai
from tqdm import tqdm


class TrainingDataGenerator:
    def __init__(self, api_key: str, proxy: str = None):
        """
        初始化训练数据生成器

        参数:
            api_key: OpenAI API密钥
            proxy: 代理服务器地址（可选），如 "http://user:pass@proxy_host:proxy_port"
        """
        # 如果需要使用代理，则通过环境变量设置
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy

        # 使用 openai 官方库
        openai.api_key = api_key

        # 设置系统提示词
        self.system_prompt = """
        你是一个专业的中国古典哲学专家。你需要：
        1. 基于用户输入的古典文本，生成相关的问答对
        2. 问题应该涵盖文本的核心思想、历史背景、作者观点等多个方面，可以参考但不限于如下问题格式：{}代表什么？周易中的{}含义是什么？请解释一下{}。{}在周易中是什么象征？周易的{}讲述了什么？{}是怎样的一个卦象？
        3. 答案要准确、全面，且基于输入得文本，同时要易于理解
        4. 每次生成20个不同的问答对
        5. 输出格式为JSON数组：[{"question": "问题", "answer": "答案"}]

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        [{"question":"周易中的师卦含义是什么？","answer":"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"}]
        """

        self.system_prompt2 = """
        你是中国古典哲学大师，尤其擅长周易的哲学解读。你需要：

        1. 你收到的都是关于周易卦象的解释，你需要整理润色，并生成用于大模型训练的内容
        2. 输出格式为JSON数组：[{"content": "师卦", "summary": "内容"}]

        示例输入：

        师卦，此卦是异卦相叠，下卦为坎，上卦为坤。“师”指军队。坎为水、为险；坤为地、为顺，喻寓兵于农。兵凶战危，用兵乃圣人不得已而为之，但它可以顺利无阻碍地解决矛盾，因为顺乎形势，师出有名，故能化凶为吉。占得此卦，对于军事上率师出征非常有利，必无灾祸。师卦是天马出群之卦，以寡伏众之象。
        师卦位于讼卦之后，《序卦》之中这样解释道：“讼必有众起，故受之以师。师者，众也。”争讼的人越来越多，以致形成了军队。

        期待结果：

        [{"content":"师卦","summary":"在周易中，师卦是一个极具深意的卦象，它由两个异卦相叠组成：下卦坎（水）和上卦坤（地）。这一卦象代表“师”，即军队，寓意着兵力和农力的结合。在这里，坎卦象征着水和险难，而坤卦象征着地和顺从，暗示着通过将军事力量安置于民间，可以在必要时顺利调动。

        师卦的核心哲学是：虽然兵力代表着危险和战争，但其使用应当是圣人不得已而为之的最后手段。在正确的情况下，军事力量可以顺应形势，将危险转化为吉祥。因此，在军事策略上，此卦象征着出征将会顺利，无灾祸。

        师卦紧随讼卦（争讼卦），在《序卦》中解释为“讼必有众起，故受之以师”。这意味着争端激化至众多人群的参与，形成了类似军队的集体力量。"}]
        """

    def read_raw_data(self, file_path: str) -> List[str]:
        """
        读取原始文本文件，并按照空行分割成一个个文本块

        参数:
            file_path: 原始数据文件路径

        返回:
            文本块列表，每个元素为一个完整的段落（用空行分割）
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按照空行分割：可用两个或以上换行符分割，也可用空行分割
        # 这里用双换行来分割(兼容多系统换行可使用 \n\n 处理)
        # 如果空行中存在 \r\n，则可用 \n\s*\n 之类的正则
        # 简单方法：split('\n\n') 亦可
        blocks = content.strip().split('\n\n')

        # 进一步去除每段首尾空白
        blocks = [block.strip() for block in blocks if block.strip()]
        return blocks
    
    # 在调用 json.loads 前，替换控制字符为其 Unicode 转义表示
    def escape_control_characters(self, s: str) -> str:
        return re.sub(r'[\x00-\x1f]', lambda match: '\\u{0:04x}'.format(ord(match.group(0))), s)
    
    def generate_refined_content(self, content: str) -> Dict[str, str]:
        """
        针对输入的关于周易卦象的解释内容生成润色后的文本，
        输出格式为JSON对象，包含 "content"（卦象名称）和 "summary"（润色后的解释）
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 或使用支持该请求的模型
                messages=[
                    {"role": "system", "content": self.system_prompt2},
                    {"role": "user", "content": (
                        "关于周易卦象的解释内容如下：\n" + content
                    )}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            refined_str = response["choices"][0]["message"]["content"]

            # 移除 Markdown 代码块标记（如果存在）
            if refined_str.startswith("```json"):
                refined_str = refined_str[len("```json"):].strip()
            if refined_str.endswith("```"):
                refined_str = refined_str[:-len("```")].strip()
            refined_str = refined_str.replace('\n', '')
            refined_content = json.loads(refined_str)
            return refined_content

        except Exception as e:
            print(f"生成润色内容时出错: {str(e)}")
            return {}

    def generate_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """
        为单个文本生成问答对

        参数:
            content: 输入文本

        返回:
            问答对列表
        """
        try:
            # 调用 openai.ChatCompletion.create 接口
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 或使用 "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"请基于以下文本生成问答对：\n{content}"}
                ],
                temperature=0.8,
                max_tokens=5000
            )

            # 从 response 中解析返回的JSON字符串
            # openai 库返回的结构一般为 response["choices"][0]["message"]["content"]
            qa_pairs_str = response["choices"][0]["message"]["content"]

            # 移除 Markdown 代码块标记
            if qa_pairs_str.startswith("```json"):
                qa_pairs_str = qa_pairs_str[len("```json"):].strip()
            if qa_pairs_str.endswith("```"):
                qa_pairs_str = qa_pairs_str[:-len("```")].strip()

            # 再将字符串转换为 JSON
            qa_pairs = json.loads(qa_pairs_str)

            # 确保返回类型为列表
            if not isinstance(qa_pairs, list):
                print(f"警告：解析后的结果不是列表，实际结果: {qa_pairs_str}")
                return []
            return qa_pairs

        except Exception as e:
            print(f"生成问答对时出错: {str(e)}")
            return []

    def save_to_csv(self, qa_pairs: List[Dict[str, str]], output_file: str):
        """
        将问答对保存到CSV文件

        参数:
            qa_pairs: 问答对列表
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["question", "answer"])
                writer.writeheader()
                writer.writerows(qa_pairs)
        except Exception as e:
            print(f"保存CSV文件时出错: {str(e)}")

    def save_refined_to_csv(self, refined_data: List[Dict[str, str]], output_file: str):
        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["content", "summary"])
                writer.writeheader()
                writer.writerows(refined_data)
        except Exception as e:
            print(f"保存润色CSV文件时出错: {str(e)}")

    def generate_dataset(self, input_file: str, output_file: str, refined_output_file: str = None, max_samples: int = None):
        """
        生成完整的训练数据集

        参数:
            input_file: 输入文件路径
            output_file: 输出文件路径
            max_samples: 最大样本数（可选）
        """
        # 读取原始数据，按照空行分块
        blocks = self.read_raw_data(input_file)

        # 如果仅调试或需要限制数量
        if max_samples:
            blocks = blocks[:max_samples]

        all_qa_pairs = []
        all_refined = []

        # 使用tqdm显示进度
        for block in tqdm(blocks, desc="生成训练数据"):
            # 生成问答对
            qa_pairs = self.generate_qa_pairs(block)
            # 追加到总列表
            all_qa_pairs.extend(qa_pairs)

            # 生成润色后的内容
            refined = self.generate_refined_content(block)
            if refined:
                all_refined.extend(refined)

            # 添加延时避免API速率限制
            time.sleep(1)

        # 保存结果到 CSV
        self.save_to_csv(all_qa_pairs, output_file)
        print(f"已生成 {len(all_qa_pairs)} 个问答对，保存至 {output_file}")

        # 如果指定了润色输出文件，则保存润色后的内容
        if refined_output_file:
            self.save_refined_to_csv(all_refined, refined_output_file)
            print(f"已生成 {len(all_refined)} 条润色内容，保存至 {refined_output_file}")

def main():
    # 配置参数
    API_KEY = ""  # 替换为你的API密钥
    INPUT_FILE = "./chatglm/data/raw_data.txt"  # 原始数据文件
    OUTPUT_FILE = "./chatglm/data/dataset.csv"  # 输出文件
    REFINED_OUTPUT_FILE = "./chatglm/data/refined_dataset.csv"  # 润色内容输出文件
    
    # 创建生成器实例
    generator = TrainingDataGenerator(
        api_key=API_KEY
    )

    # 生成数据集
    generator.generate_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        refined_output_file=REFINED_OUTPUT_FILE
    )


if __name__ == "__main__":
    main()