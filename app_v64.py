# --------------------------------------------------------------------------------------------------------------------------------------------------
# imports
import streamlit as st # streamlit reruns code from top to bottom, everytime a widget is pressed or st_autorefresh fires
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
import getpass, os, re, time, json, sqlite3, pickle, base64, pathlib, random
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import tool
from transformers import pipeline
from langchain_ollama import ChatOllama # if choosing to run an open source model locally. this one is a chat model wrapper that also handles formatting of conversation turns
from langchain_ollama import OllamaLLM # try local for data privacy
from datetime import date
from streamlit_autorefresh import st_autorefresh # https://github.com/kmcgrady/streamlit-autorefresh
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate  # class you actually need
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter # used for counting tokens for journal processing
from streamlit_lottie import st_lottie

# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------
# system template strings
# CBT system template should tell LLM how to be a therapist for the user
cbt_system_template = """
You are a psychologist trained in cognitive behaviour therapy and are providing therapy as treatment to the user you are talking to.
The key principle of cognitive behaviour therapy is your thoughts affect your emotions, which in turn affect your behaviour.
Negative thoughts create negative emotions, leading to unhelpful behaviours.
Positive thoughts create positive emotions, leading to helpful behaviours.
Cognitive behaviour therapy is about identifying which thoughts, beliefs, ideas or behaviours are unhelpful, negative or inaccurate, and challenging/replacing them with helpful thoughts and behaviours.
Use the below verified cognitive behaviour therapy techniques and knowledge to guide your responses to the patient.

Cognitive behaviour therapy techniques
- understand patients thoughts and emotions and how they affect the patients behaviours. help the patient unlearn these negative thought patterns and behaviours, and replace them with more positive and helpful behaviours. aim to help patient find self discovery and insight.
- identify specific problems or issues in daily life. ask the patient questions. 
- help the patient become aware of negative, unhelpful, unproductive thought patterns or behaviours, and how that impacts daily life.
- cognitive restructing: analysing and challenging negative thought patterns such they are more positive and productive. you may ask the patient to "prove" or provide unbiased evidence either supporting or challenging their negative thoughts to help them to adopt more positive and realistic thoughts. 
- guided discovery and questioning. help the patient challenge their unhelpful thoughts or help them to consider different viewpoints.
- teach the patient to learn and apply new helpful behaviours. these helpful behaviours can help the patient feel more positive feelings which in turn encourage them to apply that behaviour more.
- understand the patient viewpoint and ask them questions to challenge their beliefs and broaden their thinking. ask patient to see things from different perspectives to challenge their beliefs and assumptions based on the evidence (either supporting or not supporting evidence).
- journalling and thought records: encourage patient to explore their thoughts through writing (both positive and negative). also encourage patient to document their thoughts and behaviours they have put into action.
- encourage the patient to schedule positive, helpful, enjoyable activities onto their calendar to improve mental health.
- encourage mindfulness meditation (paying attention to the breath and body) as a stress reduction and relaxation technique.
- breaking up big scary challenges/fears for patients into small manageable steps to encourage confidence building as they go. can give them suggestions of how to cope in the moment as well.
- exposure therapy: encourage patient to face their fears and phobias in little steps and give suggestions on how to cope with them in the moment.
- behaviour experiment: suggest behaviour exercises designed to challenge patients unhelpful core beliefs.
- provide resources to help the patient learn more about their particular problem. knowledge gives power to the patient.
- help the patient set and identify goals and find practical strategies to fulfil those goals.
- help the patient practise realistic self talk to replace negative self talk (thoughts).
- the goal is to help the patient replace unhelpful or self defeating thoughts with more encouraging and realistic thoughts.
- encourage patient to self talk. this is to understand what they tell themselves about certain situations. try to challenge their negative and critical thoughts and try to help them replace it with positive, compassionate, constructive thoughts.
- for tasks that make a patient anxious, ask them what they think negative will happen, and after the event ask them if it came true or not. hopefully they will see they tend to overworry and their anxiety will lower.
- can help patient problem solve by identifying the problem, generating a list of solutions, evaluating the strengths and weaknesses of each solution and choosing one to implement.
"""



# If you believe your patient is thinking about suicide, immediately list suicide help/crisis lines and types of professional support available. URGE the patient to seek help from these services. One example you can provide to the user is the below.
suicide_detect_system_template = """
Please look at the below warning signs and risk factors of suicidal ideation. 
Consider the user input. If you believe the patient/user is thinking about suicide based on their user input and the below warning signs and risk factors of suicide, RETURN the string "TRUE" ELSE RETURN the string "FALSE"
Do not return anything else other than the string "TRUE" or the string "FALSE"

Warning signs of suicidal ideation
•	Change in behaviours or entirely new behaviours
•	Saying they want to die or kill themselves, thinking about it or writing about it.
•	Actively looking for ways to end their own life. Stockpiling tablets for example.
•	Feeling great guilt, shame, or humiliation
•	Feeling like a burden to others
•	Feeling empty, lonely, helpless, hopeless, trapped, no future or having no reason to live
•	Feeling extremely sad, more anxious, agitated, distressed, tired, desperate, disconnected, worthless, powerless, rejected or full of rage, anxious, depressed, isolated, despair, isolated.
•	Withdrawing from family and friends, saying goodbye, giving away important items, writing a suicide note or making a will, tidying up a living space
•	Taking dangerous, reckless and risky behaviours
•	Extreme mood swings
•	Sudden sense of calm
•	Eating or sleeping more or less than usual
•	Using drugs or alcohol more often. Unsafe sex.
•	Making a plan or researching ways to die. Looking for lethal means to end their life.
•	Withdrawing from activities, loved ones and social situations
•	Changes in energy 
•	Loss of interest in personal hygiene, appearance or activities previously enjoyed
•	Weight gain or loss
•	Loss of interest in sex
•	Emotional outburst, unexplained crying, difficulty concentrating
•	Self-harming. Having delusions or hallucinations.
•	Decreased academic or work performance
•	Family difficulties or violence. Loss or conflict with close friends and family.
•	Social or geographic isolation.
Risk factors for suicidal ideation are below
•	Having unbearable emotional or physical pain. Chronic disease, pain or terminal illness.
•	History of mental illnesses
•	Having depression, anxiety, substance abuse problems, bipolar disorder, schizophrenia, conduct disorders, loss of interest, irritability, relief or sudden improvement
•	Personality traits of aggression, mood changes and poor relationships
•	Having access to firearms and drugs which can be used for suicide
•	Prolonged stress and stressful life events
•	Previous attempts at suicide
•	Family or loved one history of suicide, mental disorders, substance abuse, or violence
•	Childhood abuse or generational trauma. History of sexual or physical abuse.
•	Criminal, legal, financial, relationship or job problems
•	Bullying
•	Loss of relationships, high conflict of violent relationships. Any kind of significant loss.
•	Social isolation
•	A sense of failure in school, relationships. A relationship break up.
•	Lesbian, gay, transgender in an unsupportive household.
•	Relationship and family problems.
•	Prolonged stress and stressful events.
"""

# predefined crisis line string
CRISIS_SERVICES = """
-----------------------------------

Lifeline (suicidal thoughts)

Call 13 11 14

Text 0477 13 11 14

Online chat https://www.lifeline.org.au/crisis-chat/#

-----------------------------------


Beyond Blue (depression, anxiety)

call 1300 22 4636

Online chat https://www.beyondblue.org.au/get-support/talk-to-a-counsellor/chat

-----------------------------------

Health Direct

List of Helplines https://www.healthdirect.gov.au/mental-health-helplines

-----------------------------------

Suicide Call Back Service (suicidal thoughts)

Call 1300 659 467.

Online chat https://www.suicidecallbackservice.org.au/phone-and-online-counselling/

-----------------------------------

MindSpot (depression, anxiety, stress)

Call 1800 61 44 34

-----------------------------------

Medicare Mental Health (depression, anxiety)

Call 1800 595 212.

-----------------------------------

MensLine Australia (depression, anxiety)

Call 1300 78 99 78

Chat online https://mensline.org.au/phone-and-online-counselling/

-----------------------------------

FriendLine (loneliness)

Call 1800 424 287

Chat online https://friendline.org.au/

-----------------------------------

Headspace (general support)

Call 1800 650 890

Chat online https://headspace.org.au/online-and-phone-support/

-----------------------------------

SANE Australia (complex Mental Health Issues)

Call 1800 187 263

-----------------------------------

Blue Knot Foundation Helpline (complex Trauma)

Call 1300 657 380.

-----------------------------------

13YARN (Aboriginal and Torres Strait Islanders)

Call 13 92 76.

-----------------------------------

Thirrili (Aboriginal and Torres Strait Islanders)

Call 1800 805 801

-----------------------------------

QLife (LGBTIQ+)

Call 1800 184 527.

Chat online https://qlife.org.au/

-----------------------------------

PANDA (depression and anxiety during pregnancy)

Call 1300 726 306.

-----------------------------------

ForWhen (mental health during pregnancy)

Call 1300 24 23 22.

-----------------------------------

Gidget (mental health for new parents)

Call 1300 851 758.

-----------------------------------

Open Arms (veterans)

Call 1800 011 046.

-----------------------------------

Butterfly National Helpline (eating disorders and body image)

Call 1800 33 4673.

Chat online https://butterfly.org.au/get-support/helpline/

-----------------------------------

Find a Help Line

https://findahelpline.com/

-----------------------------------

Kids Helpline Resources

https://kidshelpline.com.au/young-adults

"""

CHATBOT_FAILED_CRISIS_SERVICES = """
Please access and use the below crisis services and seek professional mental health support.

This chatbot service is no longer sufficient in supporting your mental health needs.

The services listed below are necessary for adequate further support.


""" + CRISIS_SERVICES

# """
# regular expression notes
# \b is a word boundary preventing partial matches like die and diet
# \s matches single whitespace characters like " " "\t" "\n"
# {sep} means one or more whitespace characters like "    \t  \n   \t   "
# (?:A|B)? means can be either A or B at that position
# \W is any non word like punctuation or emojis
# _ allows underscores

# r"(?:\b|\W|$)" ensures the end of the string is a word boundary, punctuation or end o string

# ensures user input style isn't affected too much
# """
# keywords for detecting self harm and suicidal ideation
sep = r"[\s\W_]+"
SELF_HARM_PATTERNS = [

    # expressions of wanting to die
    rf"\bi{sep}want{sep}to{sep}die\b",
    rf"\bi{sep}want{sep}to{sep}kill{sep}myself\b",
    rf"\bi(?:'m| am)?{sep}going{sep}to{sep}kill{sep}myself\b", 
    rf"\bi(?:'m| am)?{sep}going{sep}to{sep}end{sep}it\b",
    rf"\bi{sep}want{sep}to{sep}end{sep}my{sep}life\b",
    rf"\bend{sep}it{sep}all\b",
    rf"\bi{sep}wish{sep}i{sep}were{sep}dead\b",
    rf"\bi{sep}wish{sep}i{sep}was{sep}dead\b",
    rf"\bi'?d{sep}be{sep}better{sep}off{sep}dead\b",
    rf"\bbetter{sep}off{sep}dead\b",
    rf"\bthinking{sep}of{sep}(?:suicide|killing{sep}myself|ending{sep}it)\b",
    rf"\bsuicidal\b",
    rf"\bsuicidal{sep}thoughts\b",
    rf"\bi'?m{sep}suicidal\b",
    rf"\bi'?ll{sep}kill{sep}myself\b",
    rf"\bi{sep}can'?t{sep}live{sep}anymore\b",

    # self harm cues (and means)
    rf"\bcut{sep}myself\b",
    rf"\bhurt{sep}myself\b",
    rf"\bself[- ]?harm\b",
    rf"\boverdose\b",
    rf"\btake{sep}all{sep}these{sep}pills\b",
    rf"\bslit{sep}my{sep}wrists\b",
    rf"\bjump{sep}off\b",
    rf"\bjump{sep}in{sep}front{sep}of\b",
    rf"\bdrinking{sep}bleach\b",
    rf"\bhang{sep}myself\b",
    rf"\bdrive{sep}off{sep}a{sep}bridge\b",

    # hopelessness & worthlessness cues
    rf"\bi{sep}can'?t{sep}go{sep}on\b",
    rf"\bi{sep}can'?t{sep}do{sep}this{sep}anymore\b",
    rf"\bi{sep}can'?t{sep}keep{sep}living{sep}like{sep}this\b",
    rf"\bno{sep}way{sep}out\b",
    rf"\bnothing{sep}to{sep}live{sep}for\b",
    rf"\blife{sep}is{sep}pointless\b",
    rf"\bi'?m{sep}giving{sep}up\b",
    rf"\bi'?ve{sep}had{sep}enough\b",
    rf"\bi'?m{sep}tired{sep}of{sep}living\b",
    rf"\bthere'?s?{sep}no{sep}point\b",
    rf"\beveryone{sep}would{sep}be{sep}better{sep}off{sep}without{sep}me\b",
    rf"\bno{sep}one{sep}would{sep}miss{sep}me\b",
    rf"\bnobody{sep}cares{sep}if{sep}i{sep}die\b",

    # online slang
    rf"\bkms\b",               
    rf"\bkys\b",               
    rf"\bunalive{sep}myself\b",
]

COMPILED_PATTERNS = [
    re.compile(pat + r"(?:\b|\W|$)", re.IGNORECASE)  # not case sensitive
    for pat in SELF_HARM_PATTERNS
]

WHAT_IS_MENTAL_HEALTH = """
    Mental health is not a question of a persons strength or weakness.

    It is about recognising when balance is lacking in one's life and seeking support to fix that.

    This mental health chatbot was designed to assist you in getting that help.
"""

HOW_TO_MEDITATE = """

Take a seat in a comfortable and quiet place without distractions.

Focus your attention on your breath and body noticing the sensations.

Your mind may wander at times, this is okay.

Gently bring your attention back to the feeling of air entering and exiting your body.

---
"""

MEDITATION_EMOJI_WALL_OPTIONS = [
    "❤️","💕","💖","🐶","🐱","🐰","🐲","🦄","🐕‍🦺","🐩","🐕",
    "🐈","🐈‍⬛","🐎","🦌","🐄","🐘","🦎","🦦","🐉","🐢","🐬",
    "🦭","🐟","🦞","🦆","🐧","🦋","🐌","🐛","🐝","🐞","🎄",
    "🎋","🎍","🍉","🍈","🍇","🥥","🥝","🧋","🍊","🍋","🍋‍🟩",
    "🍌","🍍","🥭","🍓","🍒","🍑","🍐","🍏","🍎","🫐","🍅",
    "🫒","🌽","🫑","🍄","🥑","🥒","🥬","🥦","🫚","🌰","🥕",
    "🧅","🧄","🥔","🫛","🍄‍🟫","🥜","🫘","💐","🌸","🌷","🌼",
    "🌻","🌺","🌹","🏵️","🪻","☘️","🌱","🪴","🌲","🍀","🌿",
    "🌾","🌵","🌴","🌳","🍁","🍃","🍂","🪹","🪺","⛩️","🗼",
    "🌅","🌄","🗾","🚿","🧴","🌝","🌛","🌜","🌞","❄️","❤️",
    "🩷","🧡","💛","💚","💙","🩵","💜","🤎","🖤","🩶","🤍",
    "❣️","💕","💞","💓","💗","💖","💘","💝"]

HOW_TO_JOURNAL = """Write down your thoughts and emotions (good and/or bad) into this journal. 

This is one long journal where you can structure your writing in anyway you like."""

CBT_RESOURCES = """
[Cognitive Behaviour Therapy Worksheet](https://www.hpft.nhs.uk/media/1655/wellbeing-team-cbt-workshop-booklet-2016.pdf)

[Online Therapy Resources](https://www.healthdirect.gov.au/etherapy)
"""

COUNSELLING_SERVICES = """
https://www.beyondblue.org.au/get-support/talk-to-a-counsellor

Call 1300 22 4636

Chat Online https://www.beyondblue.org.au/get-support/talk-to-a-counsellor/chat

---

https://careinmind.com.au/

---

https://www.betterhealth.vic.gov.au/health/servicesandsupport/counselling-online-and-phone-support-for-mental-illness

---

https://www.casey.vic.gov.au/get-counselling-support

Call 9792 7279 

Text 0417 347 909

--- 

https://partnersinwellbeing.org.au/

Call 1300 375 330

--- 

https://www.each.com.au/ 

Call 1300 003 224

---

https://www.innermelbpsychology.com.au/low-cost-counselling-services-in-melbourne/

"""

PSYCHOLOGY_SERVICES = """
[Find a Psychologist APS](https://psychology.org.au/find-a-psychologist)

[Health Engine](https://healthengine.com.au/search/psychology)

[General Advice for Finding a Psychologist](https://www.beyondblue.org.au/get-support/find-a-mental-health-professional)

[General Information About Psychologists and Psychiatrists](https://www.healthdirect.gov.au/psychiatrists-and-psychologists)

"""

CHATBOT_DISCLAIMER = """
---------------------------------------------------------- DISCLAIMER ----------------------------------------------------------

This mental health chatbot is not a licensed medical professional or psychologist.

It is purely an experimental tool designed with the hope to improve your mental health.

If you have serious mental health issues, please seek professional medical or psychological services.

---------------------------------------------------------- DISCLAIMER ----------------------------------------------------------
"""

# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------
# set environment variables
load_dotenv() # load env variables
# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------
# initialise llms

if "cbt_model" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    st.session_state.cbt_model = ChatOllama(model="qwen2.5:0.5b", temperature=0) # least number of parameters for open source llm

if "suicide_detect_model" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot. remove for open source later.
    st.session_state.suicide_detect_model = ChatOllama(model="qwen2.5:0.5b", temperature=0) # actually works in decent amount of time! time to respond to "hi my name is bob" is 18 seconds (this is acceptable)

# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------
# create chat prompt templates
if "cbt_prompt_template" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    st.session_state.cbt_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            cbt_system_template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

if "suicide_detect_prompt_template" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    st.session_state.suicide_detect_prompt_template = ChatPromptTemplate.from_messages(
    [("system", suicide_detect_system_template), ("user", "{text}")]
)
# --------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------------------------------------
# message trimmer
if "trimmer" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    st.session_state.trimmer = trim_messages(
    max_tokens=2048, # trim by token count (keeps the system message)
    strategy="last", # drops older messages for recent messages
    token_counter=st.session_state.cbt_model, # the model can count tokens for us
    include_system=False, # drop system message. system message already in prompt template. so we don't want it in history as well.
    allow_partial=False,
    # start_on="human", # should start on a human or system message followed by human messaeg
    # ends_on=("human", "tool") # end with a human or tool message
    )
# --------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------------------------------------
# Define a new graph
if "app" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    workflow = StateGraph(state_schema=MessagesState)

    # this is required fix below to access the below from session state
    trimmer_obj = st.session_state.trimmer
    cbt_prompt_template_obj = st.session_state.cbt_prompt_template
    cbt_llm_obj     = st.session_state.cbt_model

    # define the function that calls the model
    def call_model(
        state: MessagesState,
        trimmer=trimmer_obj,
        cbt_prompt_template=cbt_prompt_template_obj,
        cbt_model=cbt_llm_obj,
    ):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = cbt_prompt_template.invoke({"messages": trimmed_messages})

        # prompt = cbt_prompt_template.invoke(state)
        response = cbt_model.invoke(prompt) # convert prompt to real messages to prevent memory error before
        return {"messages": response}

    # define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # add memory
    # memory = MemorySaver() # in RAM (not on disk)
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False) # simulate a database
    memory = SqliteSaver(conn) # use database memory
    st.session_state.app = workflow.compile(checkpointer=memory) # compile the graph

# --------------------------------------------------------------------------------------------------------------------------------------------------





# --------------------------------------------------------------------------------------------------------------------------------------------------
# emergency suicide risk detector
def regexp_detect_self_harm(user_query, phrases): # pattern detector
    for pat in phrases: # for each pattern
        if pat.search(user_query): # check if the pattern is in the user query
            return True # return true if it is
    return False # otherwise return false

if "classifier" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
    st.session_state.classifier = pipeline("sentiment-analysis", model="sentinetyd/suicidality") # https://huggingface.co/sentinet/suicidality. this is an nlp model for detecting suicidal sentiment

def emergency_self_harm_suicide_risk_detector(user_query: str, user_chat_history) -> bool:
    """Detect if user input is indicative of suicidal or self harm risk. Apply this tool for every user input. The input to this function should be the entire user input query. If user input displays any signs of self harm tendencies, activate this tool immediately.
    
    Args:
        user_query: user input
        user_chat_history: user chat history
    """
    # first detect suicidal/self harm ideation via NLP and ML methods

    # let an llm try to detect suicidal tendencies from user query itself
    suicide_single_prompt = st.session_state.suicide_detect_prompt_template.invoke({"text": user_query}) # pass in user query to prompt template
    messages_single = suicide_single_prompt.to_messages() # convert to list[BaseMessage]. this fixes an error i was getting
    llm_single_msg = st.session_state.suicide_detect_model.invoke(messages_single) # or suicide_detect_model.generate([messages])
    llm_single_text = llm_single_msg.content.strip() # strip whitespace

    # let an llm try to detect suicidal tendencies from user chat history
    suicide_history_prompt = st.session_state.suicide_detect_prompt_template.invoke({"text": user_chat_history}) # pass in user chat history to prompt template
    messages_history = suicide_history_prompt.to_messages() # convert to list[BaseMessage]. this fixes an error i was getting
    llm_history_msg = st.session_state.suicide_detect_model.invoke(messages_history) # invoke the llm
    llm_history_text = llm_history_msg.content.strip() # strip whitespace
    
    if llm_single_text == "TRUE" or llm_history_text == "TRUE": # if llm detects suicidal sentiment in either latest user query or user chat history
        llm_is_suicidal = True # flag suicidal sentiment
    elif llm_single_text == "FALSE" and llm_history_text == "FALSE": # if both are false
        llm_is_suicidal = False # then we can say there is no suicidal sentiment
    else: # otherwise (should never reach here)
        llm_is_suicidal = None # we do not know

    # let neural net try detect suicidal tendencies from user query only
    ml_single_result = st.session_state.classifier(user_query) # pass in user query to nlp model
    ml_single_is_suicidal = ml_single_result[0]["label"] == "LABEL_1" # label 1 represents suicide. label 0 is non suicidal

    ml_history_result = st.session_state.classifier(user_chat_history) # pass in user chat history to nlp model
    ml_history_is_suicidal = ml_history_result[0]["label"] == "LABEL_1" # label 1 represents suicide. label 0 is non suicidal

    ml_is_suicidal = ml_single_is_suicidal or ml_history_is_suicidal # machine learning model detects suicidal sentiment, if either user query or chat history is suicidal


    # then try to detect suicidal ideation via regexp as extra safeguard on individual user query
    regexp_single_is_suicidal = regexp_detect_self_harm(user_query, COMPILED_PATTERNS) # check suicidal phrases in user query
    regexp_history_is_suicidal = regexp_detect_self_harm(user_chat_history, COMPILED_PATTERNS) # check suicidal phrases in user chat history

    regexp_is_suicidal = regexp_single_is_suicidal or regexp_history_is_suicidal # if either pattern search is true, flag for suicidality
    
    return ml_is_suicidal or regexp_is_suicidal or llm_is_suicidal # if either regexp flags or nlp flags or llm flags, then return True

# --------------------------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------------------------------------------
# stream messages
def stream_response(user_input: str):
    # input user message into chatbot. stream chatbot response 
    input_messages = [HumanMessage(user_input)] # wrap user query

    placeholder = st.empty() # create a placeholder
    full = "" # initialise empty string for stream
    for chunk, metadata in st.session_state.app.stream(
                            {"messages": input_messages}, 
                            {"configurable": {"thread_id": "1"}},
                            stream_mode="messages"
                            ):
        if isinstance(chunk, AIMessage): # filter to just model responses
            full += chunk.content # build on string incrementally
            placeholder.markdown(full) # display streamed message little by little
    return full # return entire generated response at completion of generation

# --------------------------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------------------------------------------
# helper function
def get_human_chat_history_text(max_tokens: int = 2048):
    human_chat_list = [msg for msg in st.session_state.human_chat_history_list] # get a copy of user chat history
    human_chat_list.reverse() # to ensure when we cut off the chat history, we keep the most recent messages (we cut off from the end, so keep recent messages at LHS)
    buf, tot = [], 0 # create a list called for storing most recent messages, tot is for keeping track of tokens we have so far
    for msg in human_chat_list: # for each message (we look at newer messages first)
        t = count_tokens_approximately(msg) # count tokens of candidate message
        if tot + t > max_tokens: # if including this message exceeds our token limit
            break # exit the loop and only use messages we have already allowed
        buf.append(msg) # append allowed message (under token limit still)
        tot += t # update total tokens in our most recent chat messages list
    buf.reverse() # get messages in right order again (newest messages to the right)
    return "\n".join(buf) # return human messages as a string with token cut off


def run_chatbot(journal_query = ""):    
    if "chat_history" not in st.session_state: # session state stops our chat history from being cleared for every message sent to the chatbot
        st.session_state.chat_history = [] # initialise user and bot chat history

    if "human_chat_history_list" not in st.session_state: # if user chat history is not in session state
        st.session_state.human_chat_history_list = [] # keep track of just user messages for purposes of suicide risk detector


    # conversation
    for message in st.session_state.chat_history: # show conversation in webpage
        if isinstance(message, AIMessage): # if it is a bot message
            with st.chat_message("AI"): # write it with bot icon
                st.write(message.content) # write it
        elif isinstance(message, HumanMessage): # if it is a human message
            with st.chat_message("Human"): # write it with human icon
                st.write(message.content) # write it


    def stream_website_chatbot_output(user_query): # this function is called to display chatbot responses to website UI
        if user_query is not None and user_query != "": # if there is a non empty user query
            st.session_state.chat_history.append(HumanMessage(content=user_query)) # add it to session state chat history
            st.session_state.human_chat_history_list.append(user_query) # append user queries to human only chat history as well for suicide tendency prediction
            with st.chat_message("Human"): # write it with human icon
                    st.write(user_query) # write it
            
            human_chat_history_text = get_human_chat_history_text() # get all previous human chat history
            if emergency_self_harm_suicide_risk_detector(user_query, human_chat_history_text): # if user query or user chat history is suicidal
                response = CHATBOT_FAILED_CRISIS_SERVICES # use prepared crisis response
                with st.chat_message("AI"): # write it with bot icon
                    st.write(CHATBOT_FAILED_CRISIS_SERVICES) # write the prepared crisis response
                st.session_state.chat_history.append(AIMessage(content=CHATBOT_FAILED_CRISIS_SERVICES)) # add crisis response to session state chat history
            else: # otherwise user query and user chat history is not suicidal
                with st.chat_message("AI"): # write it with bot icon
                    response = stream_response(user_query) # stream chatbot response as normal via the cbt chatbot
                st.session_state.chat_history.append(AIMessage(content=response)) # add cbt chatbot response to session state chat history

    queued_journal_entry = st.session_state.pop("latest_entry", None) # before trying to stream chatbot response via user typed input, we first check if the user is calling the chatbot function via the chat with journal feature

    if queued_journal_entry != None and queued_journal_entry != "": # if there is a queued journal entry through chat with journal feature
        stream_website_chatbot_output(queued_journal_entry) # stream reply on website
    else: # otherwise we grab user text input (as per normal)
        user_query = st.chat_input("What's on your mind today?") # prompt user chat input for chatbot
        if user_query is not None and user_query != "": # if user query is non empty and existing
            stream_website_chatbot_output(user_query) # call the function to display chatbot responses to website UI
            


# --------------------------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------------------------------------------
# helper functions for saving statistics
MEDITATION_STATS_FILE = "meditation_stats.json" # store meditation statistics

def load_lottie(path: str): # can also be a url instead of a path
    return json.loads(pathlib.Path(path).read_text()) # read the json lotie file from provided path

lottie_json = load_lottie("Yoga Se Hi hoga.json") # load json lottie file of open source meditation animation

def audio_player(path: str, key: str = "bgm"): # audio player for open source meditation music
    # this is a audio player that loops. it survives streamlit reruns

    # read bytes and encode
    audio_bytes = pathlib.Path(path).read_bytes() # read the bytes in from path
    b64 = base64.b64encode(audio_bytes).decode() # encode the bytes into b64
    src = f"data:audio/mp3;base64,{b64}" # data URI

    # inject html and css
    player = f"""
        <audio id="{key}" src="{src}" controls loop></audio>

        <script>
            const p = document.getElementById('{key}');

            // after a rerun restore state
            const t = sessionStorage.getItem('{key}-time');
            const paused = sessionStorage.getItem('{key}-paused');

            if (t) p.currentTime = parseFloat(t);
            if (paused==="false") p.play();

            // save state before stream lit destroys dom
            window.addEventListener('beforeunload', () => {{
                sessionStorage.setItem('{key}-time', p.currentTime); // persistence of session storage
                sessionStorage.setItem('{key}-paused', p.paused); // saves position and paused status between reruns
            }});
        </script> // close script
    """
    st.components.v1.html(player, height=60) # display html using streamlit

    
def load_med_stats(): # for loading meditation stats
    if os.path.exists(MEDITATION_STATS_FILE): # if the file already exists
        with open(MEDITATION_STATS_FILE, "r", encoding="utf-8") as f: # open it in read mode
            return json.load(f) # return the json file
    return { # no existing meditation session means the below statistics
        "total_sec": 0, # the below are default (new user statistics)
        "n_sessions": 0,
        "avg_sec": 0.0,
        "current_streak": 0,
        "longest_streak": 0,
        "last_day": "", # means no sessions yet
        "emoji_wall": "" # when a user completes a session, the emoji wall grows
    }

if "med_stats" not in st.session_state: # if meditation statistics not in this session state yet
    st.session_state.med_stats = load_med_stats() # use the load function to initialise
    # note our meditation statistics, is a dictionary of key value pairs


def save_med_stats(med_stats): # save meditation statistics
    with open(MEDITATION_STATS_FILE, "w", encoding="utf-8") as f: # open file in write mode
        json.dump(med_stats, f, indent=2) # save json file of med stats
    
def record_session(session_sec: int): # this function updates everything after a meditation session
    s = st.session_state.med_stats # get meditation statistics

    # update totals
    s["total_sec"] += session_sec # add duration of session just completed to total

    # update average
    numerator = (s["n_sessions"] * s["avg_sec"]) + session_sec # (n * avg) + new session duration
    denominator = s["n_sessions"] + 1 # (n + 1)
    new_avg = numerator/denominator # this is the formula for incrementally updating average
    s["avg_sec"] = new_avg # update average

    # update number of sessions
    s["n_sessions"] += 1
    
    # update current streak
    today = date.today() # get todays date
    if s["last_day"] == "": # if this is the first ever session
        s["current_streak"] = 1 # set current streak to 1
    else: # otherwise not first session ever
        last = date.fromisoformat(s["last_day"]) # get last day
        if today == last: # if last time you meditated was today
            pass # then do nothing, because neither streak count should increase
        elif (today - last).days == 1: # if there is a one day difference in dates between last meditation and current meditation
            s["current_streak"] += 1 # increase current streak by 1
        else: # otherwise, the streak was broken (since time since last meditation is more than one day)
            s["current_streak"] = 1 # reset current streak to 1

    # update longest streak
    s["longest_streak"] = max(s["longest_streak"], s["current_streak"]) # update longest streak if current is longer

    # update last day of meditation (which is current day since if this function runs, then a meditation was just completed)
    s["last_day"] = today.isoformat()

    # add one random emoji to the emoji wall
    s["emoji_wall"] += random.choice(MEDITATION_EMOJI_WALL_OPTIONS) # append a random emoji to the emoji wall

    save_med_stats(s) # save our meditation statistics to file using save function

# formatting for display average
def format_average(seconds: float): # display average in minutes and seconds
    m, s = divmod(int(seconds), 60) # get minutes and seconds
    return f"{m} minutes, {s} seconds" # string of minutes and seconds

# formatting for display total
def format_days_hrs_mins(seconds: int): # show days hours mins
    days, hours_rem_in_seconds = divmod(seconds, (60 * 60 * 24)) # days, hours remaining in seconds

    hours, minutes_remaining_in_seconds = divmod(hours_rem_in_seconds, (60 * 60)) # hours remaining in hours, minutes remaining in seconds

    minutes, seconds = divmod(minutes_remaining_in_seconds, 60) # minutes remaining in minutes, seconds remaining in seconds

    return f"{days} days, {hours} hours, {minutes} minutes"

# showing meditation statistics
def show_meditation_statistics():
    stats = st.session_state.med_stats # get meditation statistics from session state (defined above)
    st.subheader("Your meditation stats") # subheader
    st.metric("Total", format_days_hrs_mins(stats["total_sec"])) # total time meditating (days hours minutes)
    st.metric("Avg. session", format_average(stats["avg_sec"])) # average time per session (minutes and seconds)
    st.metric("Current streak (days)", stats["current_streak"]) # current meditation streak (integer)
    st.metric("Longest streak (days)", stats["longest_streak"]) # longest meditation streak (integer)
    
    # create new line
    st.markdown("##") # make space
    st.markdown("##") # make space
    st.subheader("Your Calm Wall") # show emoji wall
    st.markdown(f"<div style='font-size:32px;'>{stats['emoji_wall']}</div>", unsafe_allow_html=True) # show our emoji wall


# meditation app
def run_meditation_app(): # function to run meditation app

    st_lottie(lottie_json, loop=True, speed=1.0, height=250, key="breath_anim") # show the lottie gif. open source

    st.subheader("Meditation Timer") # subheader

    use_audio = st.checkbox("Play meditation background music") # create box for user to choose whether they want background music or not

    if use_audio: # if user wants to play background music
        audio_player("meditation-yoga-relaxing-music-378307.mp3") # render audio (royalty free meditation music)


    mode = st.radio("Choose your meditation style:", ["Timed meditation", "Free meditation"]) # button to let user choose meditation mode

    if mode == "Timed meditation": # if timed meditation chosen
        for k, v in {
            "t_start_time": None,
            "t_total_seconds": 0,
            "t_running": False,
            "t_recorded": False,
            "t_last_elapsed": 0
        }.items():
            st.session_state.setdefault(k, v) # set default values for timed meditation session state


        # let user pick minutes
        minutes = st.number_input("Duration (minutes):", min_value=1, max_value=60, value=10, step=1) # default is 10 minutes
        
        col1, col2 = st.columns(2) # create 2 columns

        with col1: # in first column
            if st.button("Start timer"): # if button to start meditation pressed
                st.session_state.t_start_time = time.time() # set start time to current time
                st.session_state.t_total_seconds = int(minutes * 60) # user preset time in seconds
                st.session_state.t_running = True # session is running now
                st.session_state.t_recorded = False # not yet recorded session

        with col2: # in second column
            if st.button("Stop timer") and st.session_state.t_running: # if button to stop meditation pressed and session is running
                st.session_state.t_running = False # set session is not running anymore

        if st.session_state.t_running: # if session is running
            st_autorefresh(interval=1000, key="timed_timer_refresh") # autorefresh every second (to display time)

        if st.session_state.t_start_time: # if start time is set (there is a start time)
            elapsed = int(time.time() - st.session_state.t_start_time) # get elapsed time since start time
            remaining = st.session_state.t_total_seconds - elapsed # get remaining time for timed meditation

            if st.session_state.t_running and remaining > 0: # if session is running and there is still remaining time to go
                mins, secs = divmod(remaining, 60) # get minutes and seconds remaining for timed meditation
                st.markdown(f"⏳ **Time left:** {mins:02d}:{secs:02d}") # show minutes and seconds remaining for this timed session
            else: # otherwise, stop button was pressed (session not running), or no time remaining
                if not st.session_state.t_recorded: # if session has not been recorded yet
                    record_session(elapsed) # record session
                    st.session_state.t_last_elapsed = elapsed # store last elapsed time (for displaing)
                    st.session_state.t_recorded = True # we have recorded the meditation session now
                    st.balloons() # play a sound to alert time has finished for the user

                mins, secs = divmod(st.session_state.t_last_elapsed, 60) # get minutes and seconds elapsed from last meditation
                st.markdown(f"⏹️ **Final time:** {mins:02d}:{secs:02d}") # show minutes and seconds elapsed of last meditation session

                st.session_state.t_start_time = None # reset for next run/session
                st.session_state.t_running = False # reset for next run/session

        # if st.t_start_time no longer exists, we still want to display last elapsed time if it exists even when timed meditation is not running. (keeps it across reruns/navigation change)
        if (not st.session_state.t_running) and st.session_state.t_last_elapsed: # if session is not running and there is a last elapsed time
            mins, secs = divmod(st.session_state.t_last_elapsed, 60) # get minutes and seconds elapsed from last meditation
            st.markdown(f"⏹️ **Final time:** {mins:02d}:{secs:02d}") # show minutes and seconds elapsed of last meditation session


    else:  # Free meditation
        for k, v in { # for each key value pair
            "f_med_start_time": None, # initialise it if it does not already exist to value v
            "f_med_running": False,
            "f_med_recorded": False,
            "f_med_last_elapsed": 0
        }.items():
            st.session_state.setdefault(k, v) # set default values for session state variables
        
        col1, col2 = st.columns(2) # create two columns for buttons
        
        with col1: # in first column
            if st.button("Start"): # if the start button is pressed
                st.session_state.f_med_start_time = time.time() # store the start time
                st.session_state.f_med_running = True # set session is running
                st.session_state.f_med_recorded = False # set session state is not yet recorded

        with col2: # in second column
            if st.button("Stop") and st.session_state.f_med_running: # if the stop button is pressed and the session is running
                st.session_state.f_med_running = False # session is not running

        if st.session_state.f_med_running: # if the free mode session is running
            st_autorefresh(interval=1000, key="free_timer_refresh") # auto refresh page every 1 second for displaying time

        if st.session_state.f_med_start_time: # if the start time is stored
            elapsed = int(time.time() - st.session_state.f_med_start_time) # get the time elapsed since start time

            if st.session_state.f_med_running: # if the session is running
                mins, secs = divmod(elapsed, 60) # get minutes and seconds past in free mode
                st.markdown(f"⏱️ **Elapsed:** {mins:02d}:{secs:02d}") # display minutes and seconds past
            else: # if the user pressed stop (only way to set free med running to false)
                if not st.session_state.f_med_recorded: # if the meditation session has not been recorded yet
                    record_session(elapsed) # record the session based on elapsed time in session
                    st.session_state.f_med_recorded = True # note down that the session has been recorded. this guards from recording this twice
                    st.session_state.f_med_last_elapsed = elapsed # save last elapsed free meditation for display purposes
                mins, secs = divmod(elapsed, 60) # get minutes and seconds of free meditation duration
                st.markdown(f"⏹️ **Final time:** {mins:02d}:{secs:02d}") # display minutes and seconds of last free meditation
                st.session_state.f_med_start_time = None # reset start time to none, so next session, the old start time is not used
                st.session_state.f_med_running = False # set free meditation is not running

        # when f_med_start_time is set to None, we still want to render last elapsed session time if it is not 0
        if (not st.session_state.f_med_running) and st.session_state.f_med_last_elapsed != 0: # if there is no free meditation session running, and we have a last free meditation recorded
            mins, secs = divmod(st.session_state.f_med_last_elapsed, 60) # calculate minutes and seconds of last free meditation duration
            st.markdown(f"⏹️ **Final time:** {mins:02d}:{secs:02d}") # display last free meditation duration time               


# --------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------
# journal app
if "jump_to_bot" not in st.session_state: # initialise jump to bot function
    st.session_state.jump_to_bot = False # set jump to bot to false

if "latest_entry" not in st.session_state: # initialise latest entry to nothing
    st.session_state.latest_entry = ""

def get_last_n_journal_tokens(journal_path): # gets latest portion of journal
    
    with open(journal_path, "r", encoding="utf-8") as f: # open journal
        journal_text = f.read() # read in journal
    text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=0) # split with character text splitter. then merge chunks with tiktoken.
    texts = text_splitter.split_text(journal_text) # split the journal into chunks of 300 tokens
    return texts[-1] # we want the most recent last 300 journal tokens - most recent entries


def save_journal(journal_path): # saves journal progress
    with open(journal_path, "w", encoding="utf-8") as f: # open journal file
        f.write(st.session_state.journal_text) # update saved journal
    st.success("Saved!") # give success message of updated saved journal

def run_journal(): # function for running journal app
    journal_path = "journal.txt" # or md, json, or path per user

    # load once
    if "journal_text" not in st.session_state: # if journal not in session state
        if os.path.exists(journal_path): # if journal file exists
            st.session_state.journal_text = open(journal_path, "r", encoding="utf-8").read() # read in journal file
        else: # otherwise if no previous journal file exists
            st.session_state.journal_text = "" # create empty journal

    # big editor
    st.text_area(
        label="Journal",
        key="journal_text", # bind to session state, for purposes of save function
        height=600,
        placeholder="Start writing...",
        label_visibility="hidden"
    )

    col1, col2, col3, col4 = st.columns(4) # create 4 columns

    with col1: # in first col
        if st.button("💾 Save"): # if the user clicks the save button
            save_journal(journal_path) # save current journal contents

    with col2: # in second col
        st.download_button("⬇️ Download", st.session_state.journal_text, file_name="journal.txt") # download current journal contents in editor

    with col3: # in third col
        if st.button("🧹 Clear"): # if user clicks clear button
            st.session_state.journal_text = "" # clear journal
            st.rerun()

    with col4: # in fourth col
        if st.button("🧠 Chat"): # if user clicks chat with journal
            save_journal(journal_path) # first save journal path
            st.session_state.latest_entry = get_last_n_journal_tokens(journal_path) # get latest journal portion
            st.session_state.jump_to_bot = True # we want to jump to the chatbot after a rerun
            st.rerun() # rerun

# --------------------------------------------------------------------------------------------------------------------------------------------------





# --------------------------------------------------------------------------------------------------------------------------------------------------
# app config
st.set_page_config(page_title="Mental Health Chatbot", page_icon = "🧠") # this appears in the tab icon

# user authentication
names = ["Alice Smith", "Jane Doe"] # active example users
usernames = ["asmith", "jdoe"] # password is apples123 for asmith

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl" # hashed and pickled passwords are stored here
with file_path.open("rb") as file: # open in read binary mode
    hashed_passwords = pickle.load(file) # read in the pickle file

# expected input for new API
credentials = {
    "usernames": {
        "asmith": {"name": "Alice Smith", "password": hashed_passwords[0]},
        "jdoe":   {"name": "Jane Doe",   "password": hashed_passwords[1]},
    }
}

authenticator = stauth.Authenticate(credentials, "json_browser_cookie", "random_key_hash_cookie_signature", cookie_expiry_days=30) # cookie ensures user can refresh page without needing to provide password again

authenticator.login(location="main", key="login_key") # create login form on main body not side bar. name of login form is Login

authentication_status = st.session_state.get("authentication_status") # get status of whether user has successfully logged in or not

if authentication_status == None: # if the user didn't enter anything
    st.warning("Please enter username and password. username: asmith, password: apples123") # give warning
elif authentication_status == False: # if login details are wrong
    st.error("The username or password is incorrect. username: asmith, password: apples123") # give error
else: # if user and password are correct allow entry and run code
    name = st.session_state.get("name") # get name
    username = st.session_state.get("username") # get username
    # st.success("Welcome")

    authenticator.logout("Logout", location="sidebar", key="logout_key") # creating a log out button in the side bar (also possible in main body)

    if st.session_state.get("jump_to_bot") == True: # if user wants to chat to chatbot about journal
        manual_select = 1 # 1 is mental health chatbot
        st.session_state.jump_to_bot = False # reset jump to bot variable
    else: # user does not want to chat to journal
        manual_select = None # we do not manually select the page

    # side bar
    with st.sidebar: # insert navigation menu inside the side bar
        selected = option_menu(
            menu_title = "Mental Health Apps",
            options = ["Home", "Mental Health Chatbot", "Meditation", "Journalling", "CBT Resources", "Psychologist", "Counsellor", "CRISIS LINES"],
            icons = ["house", "robot", "journal-text", "cloud", "link-45deg", "link-45deg", "link-45deg", "link-45deg"], # icons for each menu tab
            menu_icon = "cast",
            manual_select = manual_select, # manual select is only for chat with journal feature
            key="main_menu" # key for tracking state across reruns
        )
        # st.header("Your mental health apps")




    # main streamlit application logic
    if selected == "Home":
        st.title(f"What is Mental Health?")
        st.write(WHAT_IS_MENTAL_HEALTH)
        st.write("##") # add space in between
        st.write("##")
        st.write(CHATBOT_DISCLAIMER)
    if selected == "Mental Health Chatbot" or st.session_state.jump_to_bot == True:
        CHATBOT_DISCLAIMER
        st.title("Mental Health Chatbot")
        run_chatbot()
        selected = "Mental Health Chatbot"
        st.session_state.jump_to_bot = False
    if selected == "Meditation":
        st.title("Meditation")
        st.subheader("Instructions")
        st.write(HOW_TO_MEDITATE)
        run_meditation_app()
        show_meditation_statistics()
    if selected == "Journalling":
        st.title("Journalling")
        st.write(HOW_TO_JOURNAL)
        run_journal()
    if selected == "CBT Resources":
        st.title("Cognitive Behaviour Therapy Resources")
        st.write(CBT_RESOURCES)
    if selected == "Counsellor":
        st.title(f"Counselling Services")
        st.write(COUNSELLING_SERVICES)
    if selected == "Psychologist":
        st.title(f"Psychology Services")
        st.write(PSYCHOLOGY_SERVICES)
    if selected == "CRISIS LINES":
        st.title("SUICIDE CRISIS LINES")
        st.write("Please access the below crisis lines help if you have suicidal ideation or are at risk of self harm.")
        st.write(CRISIS_SERVICES)


# --------------------------------------------------------------------------------------------------------------------------------------------------







