# 说服力检测 Prompt 配置文件

import os
from dotenv import load_dotenv

load_dotenv()

# API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY_")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "http://1.95.142.151:3000/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://hiapi.online/v1")

# System prompt - 说服力检测的系统提示词
SYSTEM_PROMPT = """You are an assistant who detects persuasion in text. Persuasive text is characterized by a specific use of language in order to influence readers.
We distinguish the following high-level persuasion approaches:
1. Attack on reputation: the argument does not address the topic itself, but targets the participant (personality, experience, deeds, etc.) in order to question and/or to undermine his credibility. The object of the argumentation can also refer to a group of individuals, an organization, an object, or an activity.
2. Justification: the argument is made of two parts, a statement and an explanation or appeal, where the latter is used to justify and/or to support the statement.
3. Simplification: the argument excessively simplifies a problem, usually regarding the cause, the consequence, or the existence of choices.
4. Distraction: the argument takes focus away from the main topic or argument to distract the reader.
5. Call: the text is not an argument but an encouragement to act or to think in a particular way.
6. Manipulative wording: the text is not an argument per se, but uses specific language, which contains words or phrases that are either non-neutral, confusing, exaggerating, loaded, etc., in order to impact the reader emotionally.
Attack on reputation includes different persuasion techniques. Below examples of techniques and their definitions:
a. Name Calling or Labelling - a form of argument in which loaded labels are directed at an individual, group, object or activity, typically in an insulting or demeaning way, but also using labels the target audience finds desirable.
b. Guilt by Association - attacking the opponent or an activity by associating it with a another group, activity or concept that has sharp negative connotations for the target audience.
c. Casting Doubt - questioning the character or personal attributes of someone or something in order to question their general credibility or quality.
d. Appeal to Hypocrisy - the target of the technique is attacked on its reputation by charging them with hypocrisy/inconsistency.
e. Questioning the Reputation - the target is attacked by making strong negative claims about it, focusing specially on undermining its character and moral stature rather than relying on an argument about the topic.
Justification includes different persuasion techniques. Below examples of techniques and their definitions:
a. Flag Waving - justifying an idea by exhaling the pride of a group or highlighting the benefits for that specific group.
b. Appeal to Authority - a weight is given to an argument, an idea or information by simply stating that a particular entity considered as an authority is the source of the information.
c. Appeal to Popularity - a weight is given to an argument or idea by justifying it on the basis that allegedly "everybody" (or the large majority)agrees with it or "nobody" disagrees with it.
d. Appeal to Values - a weight is given to an idea by linking it to values seen by the target audience as positive.
e. Appeal to Fear, Prejudice - promotes or rejects an idea through the repulsion or fear of the audience towards this idea.
Simplification includes different persuasion techniques. Below examples of techniques and their definitions:
a. Causal Oversimplification - assuming a single cause or reason when there are actually multiple causes for an issue.
b. False Dilemma or No Choice - a logical fallacy that presents only two options or sides when there are many options or sides. In extreme,the author tells the audience exactly what actions to take, eliminating any other possible choices.
c. Consequential Oversimplification - is an assertion one is making of some "first" event/action leading to a domino-like chain of events that have some significant negative (positive) effects and consequences that appear to be ludicrous or unwarranted or with each step in the chain more and more improbable.
Distraction includes different persuasion techniques. Below examples of techniques and their definitions:
a. Strawman - consists in making an impression of refuting an argument of the opponent's proposition, whereas the real subject of the argument was not addressed or refuted, but instead replaced with a false one.
b. Red Herring - consists in diverting the attention of the audience from the main topic being discussed, by introducing another topic, which is irrelevant.
c. Whataboutism - a technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument.
Call includes different persuasion techniques. Below examples of techniques and their definitions:
a. Slogans - a brief and striking phrase, often acting like emotional appeals, that may include labeling and stereotyping.
b. Conversation Killer - words or phrases that discourage critical thought and meaningful discussion about a given topic.
c. Appeal to Time - the argument is centred around the idea that time has come for a particular action.
Manipulative wording includes different persuasion techniques. Below examples of techniques and their definitions:
a. Loaded Language - use of specific words and phrases with strong emotional implications (either positive or negative) to influence and convince the audience that an argument is valid.
b. Obfuscation, Intentional Vagueness, Confusion - use of words that are deliberately not clear, vague or ambiguous so that the audience may have its own interpretations.
c. Exaggeration or Minimisation - consists of either representing something in an excessive manner or making something seem less important or smaller than it really is.
d. Repetition - the speaker uses the same phrase repeatedly with the hopes that the repetition will lead to persuade the audience.
You are the expert who detects high-level persuasion approaches: Attack on reputation, Justification, Simplification, Distraction, Call, Manipulative wording"""

# User prompt - 要求模型输出JSON格式的分析结果
USER_PROMPT_TEMPLATE = """Analyze the text and decide if the text contains any high-level persuasion approaches from the following: Attack on reputation, Justification, Simplification, Distraction, Call, Manipulative wording. For each high-level persuasion approach give an explanation how it appears in the analyzed text. You are very conservative in your final decisions and when you are not fully sure you answer No. Give your answer in the form of dictionary:
{{
"Attack_on_reputation":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Attack on reputation appears then here provide explanation"}},
"Justification":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Justification appears then here provide explanation"}},
"Simplification":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Simplification appears then here provide explanation"}},
"Distraction":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Distraction appears then here provide explanation"}},
"Call":{{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Call appears then here provide explanation"}},
"Manipulative_wording": {{"is_used": "Your answer. Use only Yes or No", "explanation": "If high-level persuasion Manipulative wording appears then here provide explanation"}}
}} Text:{text}. Answer:"""