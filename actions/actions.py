"""Custom actions"""
import os

from typing import Dict, Text, Any, List
import logging
from dateutil import parser
from datetime import date
import datetime
import sqlalchemy as sa
import actions.service as service
import actions.financial360 as financial360
import requests

# import pyodbc
from dotenv import load_dotenv

from rasa_sdk.interfaces import Action
from rasa_sdk.events import (
    SlotSet,
    EventType,
    ActionExecuted,
    SessionStarted,
    Restarted,
    FollowupAction,
    UserUtteranceReverted,
)
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.parsing import (
    parse_duckling_time_as_interval,
    parse_duckling_time,
    get_entity_details,
    parse_duckling_currency,
)

from actions.profile_db import create_database, ProfileDB

from actions.custom_forms import CustomFormValidationAction

import sendgrid
from sendgrid.helpers.mail import Mail

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from datasets import load_dataset

load_dotenv()


logger = logging.getLogger(__name__)

# The profile database is created/connected to when the action server starts
# It is populated the first time `ActionSessionStart.run()` is called .

hf_token = os.environ.get("HUGGINGFACEHUB_ADMIN_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROFILE_DB_NAME = os.environ.get("PROFILE_DB_NAME", "profile")
PROFILE_DB_URL = os.environ.get("PROFILE_DB_URL", f"sqlite:///{PROFILE_DB_NAME}.db")
ENGINE = sa.create_engine(PROFILE_DB_URL)
create_database(ENGINE, PROFILE_DB_NAME)

profile_db = ProfileDB(ENGINE)

# driver = 'ODBC Driver 17 for SQL Server'
# server = 'rasa-test.database.windows.net'
# database = 'rasa-test'
# username = 'wealthBuild'
# password = 'azureSql!23'
# cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

# cursor = cnxn.cursor()

NEXT_FORM_NAME = {
    "pay_cc": "cc_payment_form",
    "transfer_money": "transfer_money_form",
    "search_transactions": "transaction_search_form",
    "check_earnings": "transaction_search_form",
}

FORM_DESCRIPTION = {
    "cc_payment_form": "credit card payment",
    "transfer_money_form": "money transfer",
    "transaction_search_form": "transaction search",
}

dataset = load_dataset("WealthBuild/Investing_Knowledgebase", use_auth_token=hf_token)[
    "train"
]["text"]

string_dataset = " ".join(dataset).strip("\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=40, separators=["\n\n", "\n", "(?<=\. )" " ", ""]
)
chunks = text_splitter.split_text(string_dataset)

model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(
    chunks,
    model,
    metadatas=[{"sources": f"mt-{i}"} for i in range(len(chunks))],
)

retreiver = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="text-davinci-003",
    temperature=0.1,
    max_tokens=512,
    streaming=True,
    request_timeout=30,
    verbose=True,
)

llm_chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0.1,
    model="gpt-3.5-turbo-0613",
    streaming=True,
    verbose=True,
    max_tokens=512,
)

prompt_template = """ You’re an expert financial coach.  You have been helping people with financial planning for 20 years. Your task is now to give the best advice if someone ask question. If you don't know the answer that's fine.  The question given below will be related to financial term. Please generate answers in less than 60 words if possible. You must ALWAYS ask questions after you give the advice. Limit questions to 2 maximum to better understand how to best answer the user questions. Is that understood?
No Hallucinations, please quote source when possible.

Question: {question}

Answer: """

prompt_template_affirm = """You’re an expert financial coach.  You have been helping people with financial planning for 20 years. Your task is now to give the best advice using the following pieces of context at the end. If you don't know the answer that's fine. The Texts will be a sentence or paragraph about some financial term and ask user if they want to know more about something within the context. Please generate answers based on this context and Texts with more details in less than 150 words if possible. You must ALWAYS ask questions after you give the advice. Limit questions to 2 maximum to better understand how to best answer the user questions. Is that understood? No Hallucinations, please quote source when possible.
{context}
Texts: {question} 

Helpful Answer with questions at the end: """

prompt_template_intent = """You’re an expert financial coach.  You have been helping people with financial planning for 20 years. Your task is now to give the best advice using the following pieces of context at the end. If you don't know the answer that's fine. There will be Bot Message and User Message. The Bot Message will be a sentence or paragraph about some financial term and User Message will be a question or just simple word or sentence or paragraph. Please, you must figure out the Bot Message and User Message are in same context or not. If User Message is a word/sentence/paragraph related to the Bot Message then  Please generate answers based on this context and Bot Message and User Message with more details in less than 300 tokens.  If User Message is something off topic to the Bot Message then you must generate answers based on User Message only with more details in less than 300 tokens. ask user if they want to know more about something within the context. You must ALWAYS ask questions after you give the advice. Limit questions to 2 maximum to better understand how to best answer the user questions. Is that understood? No Hallucinations, please quote source when possible.
{context}

{question} 

Helpful Answer with questions at the end: """


def retrieval_qa_chain_with_affirm(message):
    prompt = PromptTemplate(
        template=prompt_template_affirm,
        input_variables=["context", "question"],
    )
    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm_chat,
        chain_type="stuff",
        retriever=retreiver,
        chain_type_kwargs=chain_type_kwargs,
    )

    result = qa.run(message)
    return result


def retrieval_qa_chain_with_intent(message):
    prompt = PromptTemplate(
        template=prompt_template_intent,
        input_variables=["context", "question"],
    )
    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm_chat,
        chain_type="stuff",
        retriever=retreiver,
        chain_type_kwargs=chain_type_kwargs,
    )

    result = qa.run(message)
    return result


def llm_chain(message):
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    result = llm_chain.run(question=message)
    return result


class ActionPayCC(Action):
    """Pay credit card."""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_pay_cc"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Executes the action"""

        slots = {
            "AA_CONTINUE_FORM": None,
            "zz_confirm_form": None,
            "credit_card": None,
            "account_type": None,
            "amount-of-money": None,
            "time": None,
            "time_formatted": None,
            "start_time": None,
            "end_time": None,
            "start_time_formatted": None,
            "end_time_formatted": None,
            "grain": None,
            "number": None,
        }

        if tracker.get_slot("zz_confirm_form") == "yes":
            credit_card = tracker.get_slot("credit_card")
            amount_of_money = float(tracker.get_slot("amount-of-money"))
            amount_transferred = float(tracker.get_slot("amount_transferred"))
            profile_db.pay_off_credit_card(
                tracker.sender_id, credit_card, amount_of_money
            )

            dispatcher.utter_message(response="utter_cc_pay_scheduled")

            slots["amount_transferred"] = amount_transferred + amount_of_money
        else:
            dispatcher.utter_message(response="utter_cc_pay_cancelled")

        return [SlotSet(slot, value) for slot, value in slots.items()]


class ValidatePayCCForm(CustomFormValidationAction):
    """Validates Slots of the cc_payment_form"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "validate_cc_payment_form"

    def amount_from_balance(
        self, dispatcher, tracker, credit_card_name, balance_type
    ) -> Dict[Text, Any]:
        amount_balance = profile_db.get_credit_card_balance(
            tracker.sender_id, credit_card_name, balance_type
        )
        account_balance = profile_db.get_account_balance(tracker.sender_id)
        if account_balance < float(amount_balance):
            dispatcher.utter_message(response="utter_insufficient_funds")
            return {"amount-of-money": None}
        return {
            "amount-of-money": f"{amount_balance:.2f}",
            "payment_amount_type": f"(your {balance_type})",
        }

    async def validate_amount_of_money(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'amount-of-money' slot"""
        if not value:
            return {"amount-of-money": None}

        account_balance = profile_db.get_account_balance(tracker.sender_id)
        # check if user asked to pay the full or the minimum balance
        if type(value) is str:
            credit_card_name = tracker.get_slot("credit_card")
            if credit_card_name:
                credit_card = profile_db.get_credit_card(
                    tracker.sender_id, credit_card_name
                )
            else:
                credit_card = None
            balance_types = profile_db.list_balance_types()
            if value and value.lower() in balance_types:
                balance_type = value.lower()
                if not credit_card:
                    dispatcher.utter_message(
                        f"I see you'd like to pay the {balance_type}."
                    )
                    return {"amount-of-money": balance_type}
                slots_to_set = self.amount_from_balance(
                    dispatcher, tracker, credit_card_name, balance_type
                )
                if float(slots_to_set.get("amount-of-money")) == 0:
                    dispatcher.utter_message(
                        response="utter_nothing_due", **slots_to_set
                    )
                    return {
                        "amount-of-money": None,
                        "credit_card": None,
                        "payment_amount_type": None,
                    }
                return slots_to_set

        try:
            entity = get_entity_details(
                tracker, "amount-of-money"
            ) or get_entity_details(tracker, "number")
            amount_currency = parse_duckling_currency(entity)
            if not amount_currency:
                raise TypeError
            if account_balance < float(amount_currency.get("amount-of-money")):
                dispatcher.utter_message(response="utter_insufficient_funds")
                return {"amount-of-money": None}
            return amount_currency
        except (TypeError, AttributeError):
            pass

        dispatcher.utter_message(response="utter_no_payment_amount")
        return {"amount-of-money": None}

    async def validate_credit_card(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'credit_card' slot"""
        if value and value.lower() in profile_db.list_credit_cards(tracker.sender_id):
            amount = tracker.get_slot("amount-of-money")
            credit_card_slot = {"credit_card": value.title()}
            balance_types = profile_db.list_balance_types()
            if amount and amount.lower() in balance_types:
                updated_amount = self.amount_from_balance(
                    dispatcher, tracker, value.lower(), amount
                )
                if float(updated_amount.get("amount-of-money")) == 0:
                    dispatcher.utter_message(
                        response="utter_nothing_due", **updated_amount
                    )
                    return {
                        "amount-of-money": None,
                        "credit_card": None,
                        "payment_amount_type": None,
                    }
                account_balance = profile_db.get_account_balance(tracker.sender_id)
                if account_balance < float(updated_amount.get("amount-of-money")):
                    dispatcher.utter_message(
                        response="utter_insufficient_funds_specific", **updated_amount
                    )
                    return {"amount-of-money": None}
                return {**credit_card_slot, **updated_amount}
            return credit_card_slot

        dispatcher.utter_message(response="utter_no_creditcard")
        return {"credit_card": None}

    async def explain_credit_card(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Explains 'credit_card' slot"""
        dispatcher.utter_message("You have the following credits cards:")
        for credit_card in profile_db.list_credit_cards(tracker.sender_id):
            current_balance = profile_db.get_credit_card_balance(
                tracker.sender_id, credit_card
            )
            dispatcher.utter_message(
                response="utter_credit_card_balance",
                **{
                    "credit_card": credit_card.title(),
                    "amount-of-money": f"{current_balance:.2f}",
                },
            )
        return {}

    async def validate_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'time' slot"""
        timeentity = get_entity_details(tracker, "time")
        parsedtime = timeentity and parse_duckling_time(timeentity)
        if not parsedtime:
            dispatcher.utter_message(response="utter_no_transactdate")
            return {"time": None}
        return parsedtime

    async def validate_zz_confirm_form(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'zz_confirm_form' slot"""
        if value in ["yes", "no"]:
            return {"zz_confirm_form": value}

        return {"zz_confirm_form": None}


class ActionTransactionSearch(Action):
    """Searches for a transaction"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_transaction_search"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Executes the action"""
        slots = {
            "AA_CONTINUE_FORM": None,
            "zz_confirm_form": None,
            "time": None,
            "time_formatted": None,
            "start_time": None,
            "end_time": None,
            "start_time_formatted": None,
            "end_time_formatted": None,
            "grain": None,
            "search_type": None,
            "vendor_name": None,
        }

        if tracker.get_slot("zz_confirm_form") == "yes":
            search_type = tracker.get_slot("search_type")
            deposit = search_type == "deposit"
            vendor = tracker.get_slot("vendor_name")
            vendor_name = f" at {vendor.title()}" if vendor else ""
            start_time = parser.isoparse(tracker.get_slot("start_time"))
            end_time = parser.isoparse(tracker.get_slot("end_time"))
            transactions = profile_db.search_transactions(
                tracker.sender_id,
                start_time=start_time,
                end_time=end_time,
                deposit=deposit,
                vendor=vendor,
            )

            aliased_transactions = transactions.subquery()
            total = profile_db.session.query(
                sa.func.sum(aliased_transactions.c.amount)
            )[0][0]
            if not total:
                total = 0
            numtransacts = transactions.count()
            slotvars = {
                "total": f"{total:.2f}",
                "numtransacts": numtransacts,
                "start_time_formatted": tracker.get_slot("start_time_formatted"),
                "end_time_formatted": tracker.get_slot("end_time_formatted"),
                "vendor_name": vendor_name,
            }

            dispatcher.utter_message(
                response=f"utter_searching_{search_type}_transactions",
                **slotvars,
            )
            dispatcher.utter_message(
                response=f"utter_found_{search_type}_transactions", **slotvars
            )
        else:
            dispatcher.utter_message(response="utter_transaction_search_cancelled")

        return [SlotSet(slot, value) for slot, value in slots.items()]


class ActionMxTransactionSearch(Action):
    """Searches for a mx transaction"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_mx_transactions_search"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Executes the action"""
        slots = {
            "AA_CONTINUE_FORM": None,
            "zz_confirm_form": None,
            "time": None,
            "time_formatted": None,
            "start_time": None,
            "end_time": None,
            "start_time_formatted": None,
            "end_time_formatted": None,
            "grain": None,
            "search_type": None,
            "vendor_name": None,
        }
        transaction = ""
        startDate = ""
        time = tracker.get_slot("time")
        mxUser = tracker.get_slot("mxUser")
        today = date.today()

        if time == "week":
            startDate = today - datetime.timedelta(days=7)
        if time == "month":
            startDate = today - datetime.timedelta(days=30)
        transaction = service.get_mx_transaction(mxUser, startDate, today)

        text = (
            f"Here are your 3 most recent transactions."
            f"category: {transaction[0].category} | description: {transaction[0].description} | amount: ${transaction[0].amount} at {transaction[0].createdAt}"
            f"category: {transaction[1].category} | description: {transaction[1].description} | amount: ${transaction[1].amount} at {transaction[1].createdAt}"
            f"category: {transaction[2].category} | description: {transaction[2].description} | amount: ${transaction[2].amount} at {transaction[2].createdAt}"
        )
        dispatcher.utter_message(text=text)
        # dispatcher.utter_message(response="utter_mx_transactions", time=time, mx_transaction=transaction)
        return [SlotSet("time", None)]


class ValidateTransactionSearchForm(CustomFormValidationAction):
    """Validates Slots of the transaction_search_form"""

    def name(self) -> Text:
        """Unique identifier of the form"""
        return "validate_transaction_search_form"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Custom validates the filled slots"""
        events = await super().run(dispatcher, tracker, domain)

        # For 'spend' type transactions we need to know the vendor_name
        search_type = tracker.get_slot("search_type")
        if search_type == "spend":
            vendor_name = tracker.get_slot("vendor_name")
            if not vendor_name:
                events.append(SlotSet("requested_slot", "vendor_name"))

        return events

    async def validate_search_type(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'search_type' slot"""
        if value in ["spend", "deposit"]:
            return {"search_type": value}

        return {"search_type": None}

    async def validate_vendor_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'vendor_name' slot"""
        if value and value.lower() in profile_db.list_vendors():
            return {"vendor_name": value}

        dispatcher.utter_message(response="utter_no_vendor_name")
        return {"vendor_name": None}

    async def validate_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'time' slot"""
        timeentity = get_entity_details(tracker, "time")
        parsedinterval = timeentity and parse_duckling_time_as_interval(timeentity)
        if not parsedinterval:
            dispatcher.utter_message(response="utter_no_transactdate")
            return {"time": None}

        return parsedinterval


class ActionTransferMoney(Action):
    """Transfers Money."""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_transfer_money"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the action"""
        slots = {
            "AA_CONTINUE_FORM": None,
            "zz_confirm_form": None,
            "PERSON": None,
            "amount-of-money": None,
            "number": None,
        }

        if tracker.get_slot("zz_confirm_form") == "yes":
            amount_of_money = float(tracker.get_slot("amount-of-money"))
            from_account_number = profile_db.get_account_number(
                profile_db.get_account_from_session_id(tracker.sender_id)
            )
            to_account_number = profile_db.get_account_number(
                profile_db.get_recipient_from_name(
                    tracker.sender_id, tracker.get_slot("PERSON")
                )
            )
            profile_db.transact(
                from_account_number,
                to_account_number,
                amount_of_money,
            )

            dispatcher.utter_message(response="utter_transfer_complete")

            amount_transferred = float(tracker.get_slot("amount_transferred"))
            slots["amount_transferred"] = amount_transferred + amount_of_money
        else:
            dispatcher.utter_message(response="utter_transfer_cancelled")

        return [SlotSet(slot, value) for slot, value in slots.items()]


class ValidateTransferMoneyForm(CustomFormValidationAction):
    """Validates Slots of the transfer_money_form"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "validate_transfer_money_form"

    async def validate_PERSON(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'PERSON' slot"""
        # It is possible that both Spacy & DIET extracted the PERSON
        # Just pick the first one
        if isinstance(value, list):
            value = value[0]

        name = value.lower() if value else None
        known_recipients = profile_db.list_known_recipients(tracker.sender_id)
        first_names = [name.split()[0] for name in known_recipients]
        if name is not None and name in known_recipients:
            return {"PERSON": name.title()}

        if name in first_names:
            index = first_names.index(name)
            fullname = known_recipients[index]
            return {"PERSON": fullname.title()}

        dispatcher.utter_message(response="utter_unknown_recipient", PERSON=value)
        return {"PERSON": None}

    async def explain_PERSON(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Explains 'PERSON' slot"""
        recipients = profile_db.list_known_recipients(tracker.sender_id)
        formatted_recipients = "\n" + "\n".join(
            [f"- {recipient.title()}" for recipient in recipients]
        )
        dispatcher.utter_message(
            response="utter_recipients",
            formatted_recipients=formatted_recipients,
        )
        return {}

    async def validate_amount_of_money(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'amount-of-money' slot"""
        account_balance = profile_db.get_account_balance(tracker.sender_id)
        try:
            entity = get_entity_details(
                tracker, "amount-of-money"
            ) or get_entity_details(tracker, "number")
            amount_currency = parse_duckling_currency(entity)
            if not amount_currency:
                raise TypeError
            if account_balance < float(amount_currency.get("amount-of-money")):
                dispatcher.utter_message(response="utter_insufficient_funds")
                return {"amount-of-money": None}
            return amount_currency
        except (TypeError, AttributeError):
            dispatcher.utter_message(response="utter_no_payment_amount")
            return {"amount-of-money": None}

    async def validate_zz_confirm_form(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validates value of 'zz_confirm_form' slot"""
        if value in ["yes", "no"]:
            return {"zz_confirm_form": value}

        return {"zz_confirm_form": None}


class ActionShowBalance(Action):
    """Shows the balance of bank or credit card accounts"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_show_balance"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        account_type = tracker.get_slot("account_type")

        if account_type == "credit":
            # show credit card balance
            credit_card = tracker.get_slot("credit_card")
            available_cards = profile_db.list_credit_cards(tracker.sender_id)

            if credit_card and credit_card.lower() in available_cards:
                current_balance = profile_db.get_credit_card_balance(
                    tracker.sender_id, credit_card
                )
                dispatcher.utter_message(
                    response="utter_credit_card_balance",
                    **{
                        "credit_card": credit_card.title(),
                        "credit_card_balance": f"{current_balance:.2f}",
                    },
                )
            else:
                for credit_card in profile_db.list_credit_cards(tracker.sender_id):
                    current_balance = profile_db.get_credit_card_balance(
                        tracker.sender_id, credit_card
                    )
                    dispatcher.utter_message(
                        response="utter_credit_card_balance",
                        **{
                            "credit_card": credit_card.title(),
                            "credit_card_balance": f"{current_balance:.2f}",
                        },
                    )
        else:
            # show bank account balance
            account_balance = profile_db.get_account_balance(tracker.sender_id)
            amount = tracker.get_slot("amount_transferred")
            if amount:
                amount = float(tracker.get_slot("amount_transferred"))
                init_account_balance = account_balance + amount
                dispatcher.utter_message(
                    response="utter_changed_account_balance",
                    init_account_balance=f"{init_account_balance:.2f}",
                    account_balance=f"{account_balance:.2f}",
                )
            else:
                dispatcher.utter_message(
                    response="utter_account_balance",
                    init_account_balance=f"{account_balance:.2f}",
                )

        events = []
        active_form_name = tracker.active_form.get("name")
        if active_form_name:
            # keep the tracker clean for the predictions with form switch stories
            events.append(UserUtteranceReverted())
            # trigger utter_ask_{form}_AA_CONTINUE_FORM, by making it the requested_slot
            events.append(SlotSet("AA_CONTINUE_FORM", None))
            # avoid that bot goes in listen mode after UserUtteranceReverted
            events.append(FollowupAction(active_form_name))

        return events


class ActionShowRecipients(Action):
    """Lists the contents of then known_recipients slot"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_show_recipients"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        recipients = profile_db.list_known_recipients(tracker.sender_id)
        formatted_recipients = "\n" + "\n".join(
            [f"- {recipient.title()}" for recipient in recipients]
        )
        dispatcher.utter_message(
            response="utter_recipients",
            formatted_recipients=formatted_recipients,
        )

        events = []
        active_form_name = tracker.active_form.get("name")
        if active_form_name:
            # keep the tracker clean for the predictions with form switch stories
            events.append(UserUtteranceReverted())
            # trigger utter_ask_{form}_AA_CONTINUE_FORM, by making it the requested_slot
            events.append(SlotSet("AA_CONTINUE_FORM", None))
            # # avoid that bot goes in listen mode after UserUtteranceReverted
            events.append(FollowupAction(active_form_name))

        return events


class ActionShowTransferCharge(Action):
    """Lists the transfer charges"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_show_transfer_charge"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        dispatcher.utter_message(response="utter_transfer_charge")

        events = []
        active_form_name = tracker.active_form.get("name")
        if active_form_name:
            # keep the tracker clean for the predictions with form switch stories
            events.append(UserUtteranceReverted())
            # trigger utter_ask_{form}_AA_CONTINUE_FORM, by making it the requested_slot
            events.append(SlotSet("AA_CONTINUE_FORM", None))
            # # avoid that bot goes in listen mode after UserUtteranceReverted
            events.append(FollowupAction(active_form_name))

        return events


class ActionSessionStart(Action):
    """Executes at start of session"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_session_start"

    @staticmethod
    def _slot_set_events_from_tracker(
        tracker: "Tracker",
    ) -> List["SlotSet"]:
        """Fetches SlotSet events from tracker and carries over keys and values"""

        # when restarting most slots should be reset
        relevant_slots = ["currency"]

        return [
            SlotSet(
                key=event.get("name"),
                value=event.get("value"),
            )
            for event in tracker.events
            if event.get("event") == "slot" and event.get("name") in relevant_slots
        ]

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        """Executes the custom action"""
        # the session should begin with a `session_started` event
        events = [SessionStarted()]

        events.extend(self._slot_set_events_from_tracker(tracker))

        # create a mock profile by populating database with values specific to tracker.sender_id
        profile_db.populate_profile_db(tracker.sender_id)
        currency = profile_db.get_currency(tracker.sender_id)

        # initialize slots from mock profile
        events.append(SlotSet("currency", currency))

        # add `action_listen` at the end
        events.append(ActionExecuted("action_listen"))

        return events


class ActionRestart(Action):
    """Executes after restart of a session"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_restart"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        """Executes the custom action"""
        return [Restarted(), FollowupAction("action_session_start")]


class ActionAskTransactionSearchFormConfirm(Action):
    """Asks for the 'zz_confirm_form' slot of 'transaction_search_form'

    A custom action is used instead of an 'utter_ask' response because a different
    question is asked based on 'search_type' and 'vendor_name' slots.
    """

    def name(self) -> Text:
        return "action_ask_transaction_search_form_zz_confirm_form"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        search_type = tracker.get_slot("search_type")
        vendor_name = tracker.get_slot("vendor_name")
        start_time_formatted = tracker.get_slot("start_time_formatted")
        end_time_formatted = tracker.get_slot("end_time_formatted")

        if vendor_name:
            vendor_name = f" with {vendor_name}"
        else:
            vendor_name = ""
        if search_type == "spend":
            text = (
                f"Do you want to search for transactions{vendor_name} between "
                f"{start_time_formatted} and {end_time_formatted}?"
            )
        elif search_type == "deposit":
            text = (
                f"Do you want to search deposits made to your account between "
                f"{start_time_formatted} and {end_time_formatted}?"
            )
        buttons = [
            {"payload": "/affirm", "title": "Yes"},
            {"payload": "/deny", "title": "No"},
        ]

        dispatcher.utter_message(text=text, buttons=buttons)

        return []


class ActionSwitchFormsAsk(Action):
    """Asks to switch forms"""

    def name(self) -> Text:
        return "action_switch_forms_ask"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        active_form_name = tracker.active_form.get("name")
        intent_name = tracker.latest_message["intent"]["name"]
        next_form_name = NEXT_FORM_NAME.get(intent_name)

        if (
            active_form_name not in FORM_DESCRIPTION.keys()
            or next_form_name not in FORM_DESCRIPTION.keys()
        ):
            logger.debug(
                f"Cannot create text for `active_form_name={active_form_name}` & "
                f"`next_form_name={next_form_name}`"
            )
            next_form_name = None
        else:
            text = (
                f"We haven't completed the {FORM_DESCRIPTION[active_form_name]} yet. "
                f"Are you sure you want to switch to {FORM_DESCRIPTION[next_form_name]}?"
            )
            buttons = [
                {"payload": "/affirm", "title": "Yes"},
                {"payload": "/deny", "title": "No"},
            ]
            dispatcher.utter_message(text=text, buttons=buttons)
        return [SlotSet("next_form_name", next_form_name)]


class ActionSwitchFormsDeny(Action):
    """Does not switch forms"""

    def name(self) -> Text:
        return "action_switch_forms_deny"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        active_form_name = tracker.active_form.get("name")

        if active_form_name not in FORM_DESCRIPTION.keys():
            logger.debug(
                f"Cannot create text for `active_form_name={active_form_name}`."
            )
        else:
            text = f"Ok, let's continue with the {FORM_DESCRIPTION[active_form_name]}."
            dispatcher.utter_message(text=text)

        return [SlotSet("next_form_name", None)]


class ActionSwitchFormsAffirm(Action):
    """Switches forms"""

    def name(self) -> Text:
        return "action_switch_forms_affirm"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        active_form_name = tracker.active_form.get("name")
        next_form_name = tracker.get_slot("next_form_name")

        if (
            active_form_name not in FORM_DESCRIPTION.keys()
            or next_form_name not in FORM_DESCRIPTION.keys()
        ):
            logger.debug(
                f"Cannot create text for `active_form_name={active_form_name}` & "
                f"`next_form_name={next_form_name}`"
            )
        else:
            text = (
                f"Great. Let's switch from the {FORM_DESCRIPTION[active_form_name]} "
                f"to {FORM_DESCRIPTION[next_form_name]}. "
                f"Once completed, you will have the option to switch back."
            )
            dispatcher.utter_message(text=text)

        return [
            SlotSet("previous_form_name", active_form_name),
            SlotSet("next_form_name", None),
        ]


class ActionSwitchBackAsk(Action):
    """Asks to switch back to previous form"""

    def name(self) -> Text:
        return "action_switch_back_ask"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        """Executes the custom action"""
        previous_form_name = tracker.get_slot("previous_form_name")

        if previous_form_name not in FORM_DESCRIPTION.keys():
            logger.debug(
                f"Cannot create text for `previous_form_name={previous_form_name}`"
            )
            previous_form_name = None
        else:
            text = (
                f"Would you like to go back to the "
                f"{FORM_DESCRIPTION[previous_form_name]} now?."
            )
            buttons = [
                {"payload": "/affirm", "title": "Yes"},
                {"payload": "/deny", "title": "No"},
            ]
            dispatcher.utter_message(text=text, buttons=buttons)

        return [SlotSet("previous_form_name", None)]


class AskForLastNameAction(Action):
    def name(self) -> Text:
        return "action_ask_last_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        first_name = tracker.get_slot("first_name")
        dispatcher.utter_message(text=f"So {first_name}, what is your last name?")

        return []


class ActionCheckMxAccount(Action):
    def name(self) -> Text:
        return "action_check_mxaccount"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        mxUser = tracker.get_slot("mxUser")
        result = []
        if mxUser:
            result = [FollowupAction("action_mx_accountbalance")]
        else:
            dispatcher.utter_message(response="utter_mx_noAccount")
        return result


class ActionMxBalance(Action):
    def name(self) -> Text:
        return "action_mx_accountbalance"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        userName = tracker.get_slot("userName")
        mxUser = tracker.get_slot("mxUser")
        balance = service.get_mx_balance(mxUser)

        dispatcher.utter_message(
            response="utter_mx_accountbalance", mx_balance=balance, userName=userName
        )

        return [FollowupAction("utter_afterSession")]


class ActionSavingsGoalAmount(Action):
    def name(self) -> Text:
        return "action_savings_goal_amount"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        savings_amount_of_money = tracker.get_slot("savings_amount_of_money")
        savings_timeline = tracker.get_slot("savings_timeline")
        savings_goal_amount = int(int(savings_amount_of_money) / int(savings_timeline))
        dispatcher.utter_message(
            response="utter_savings_goal", savings_goal_amount=savings_goal_amount
        )
        dispatcher.utter_message(
            response="utter_savings_takeaction", savings_goal_amount=savings_goal_amount
        )

        return []


class ActionGreet(Action):
    def name(self) -> Text:
        return "action_greet"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        userName = tracker.get_slot("userName")

        if userName:
            result = [FollowupAction("action_start_session_afterGreet")]
        else:
            result = [FollowupAction("action_start_nickname_form")]
        return result


class ActionAskNickname(Action):
    def name(self) -> Text:
        return "action_start_nickname_form"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_greet")
        return []


class ActionStartSessionAfterGreet(Action):
    def name(self) -> Text:
        return "action_start_session_afterGreet"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        userName = tracker.get_slot("userName")
        userId = tracker.get_slot("userId")
        dispatcher.utter_message(
            response="utter_greet_user", userName=userName, userId=userId
        )
        return []


class ActionSavingsForm(Action):
    """Executes after restart of a session"""

    def name(self) -> Text:
        """Unique identifier of the action"""
        return "action_savings_form"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        """Executes the custom action"""
        return [
            SlotSet("savings_amount_of_money", None),
            SlotSet("savings_timeline", None),
            FollowupAction("savings_form"),
        ]


class ActionFinancial360(Action):
    def name(self) -> Text:
        return "action_financial360"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        isMarried = tracker.get_slot("isMarried")
        haveChildren = tracker.get_slot("haveChildren")
        haveBudget = tracker.get_slot("haveBudget")
        haveSavings = tracker.get_slot("haveSavings")
        checkCredit = tracker.get_slot("checkCredit")
        if isMarried == "yes":
            isMarriedAnswer = financial360.isMarried.yes
        elif isMarried == "no":
            isMarriedAnswer = financial360.isMarried.no
        if haveChildren == "yes":
            haveChildrenAnswer = financial360.haveChildren.yes
        elif haveChildren == "no":
            haveChildrenAnswer = financial360.haveChildren.no
        if haveBudget == "yes":
            haveBudgetAnswer = financial360.haveBudget.yes
        elif haveBudget == "no":
            haveBudgetAnswer = financial360.haveBudget.no
        if haveSavings == "yes":
            haveSavingsAnswer = financial360.haveSavings.yes
        elif haveSavings == "no":
            haveSavingsAnswer = financial360.haveSavings.no
        if checkCredit == "yes":
            checkCreditAnswer = financial360.checkCredit.yes
        elif checkCredit == "no":
            checkCreditAnswer = financial360.checkCredit.no

        email = tracker.get_slot("email")
        # from_email = Email("jhp1818@gmail.com")  # Change to your verified sender
        # to_email = To(email)  # Change to your recipient
        subject = "Here is your financial recommendation by Yoli"
        # content = Content("text/plain", "go to wealthbuild.ai for more information")
        # mail = Mail(from_email, to_email, subject, content)
        message = Mail(from_email=os.environ.get("ADMIN_EMAIL"), to_emails=email)
        message.dynamic_template_data = {
            "subject": subject,
            "isMarriedQuestion": financial360.isMarried.question,
            "isMarriedAnswer": isMarriedAnswer,
            "haveChildrenQuestion": financial360.haveChildren.question,
            "haveChildrenAnswer": haveChildrenAnswer,
            "haveBudgetQuestion": financial360.haveBudget.question,
            "haveBudgetAnswer": haveBudgetAnswer,
            "haveSavingsQuestion": financial360.haveSavings.question,
            "haveSavingsAnswer": haveSavingsAnswer,
            "checkCreditQuestion": financial360.checkCredit.question,
            "checkCreditAnswer": checkCreditAnswer,
        }
        message.template_id = os.environ.get("TEMPLATE_ID")

        # Get a JSON-ready representation of the Mail object
        # mail_json = mail.get()
        try:
            # Send an HTTP POST request to /mail/send
            sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
            # response = sg.client.mail.send.post(request_body=mail_json)
            response = sg.send(message)
            print(response.status_code)
            print(response.headers)
            print(response.body)

        except Exception as e:
            print(e)

        dispatcher.utter_message(response="utter_financial360")
        return []


class ActionLifeInsurance(Action):
    def name(self) -> Text:
        return "action_lifeInsurance_session"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        userName = tracker.get_slot("userName")

        if userName:
            dispatcher.utter_message(
                response="utter_lifeInsurance_session", userName=userName
            )
        else:
            dispatcher.utter_message(
                response="utter_lifeInsurance_session", userName=""
            )
        return []


class ActionLifeInsuranceEstimate(Action):
    def name(self) -> Text:
        return "action_lifeInsurance_estimate"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        mortgage_amount = tracker.get_slot("mortgage_amount")
        annual_salary = tracker.get_slot("annual_salary")
        cumulative_debts = tracker.get_slot("cumulative_debts")
        how_many_children = tracker.get_slot("how_many_children")
        estimate = (
            int(mortgage_amount)
            + (int(annual_salary) * 10)
            + int(cumulative_debts)
            + (75000 * int(how_many_children))
        )
        dispatcher.utter_message(
            response="utter_lifeInsurance_estimate", estimate=estimate
        )

        return []


class ActionLifeInsurance(Action):
    def name(self) -> Text:
        return "action_retirement_unhappy_session"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        intent_name = tracker.latest_message["intent"]["name"]
        result = []
        if intent_name == "deny":
            dispatcher.utter_message(response="utter_retirement_unhappy_deny1")
            dispatcher.utter_message(response="utter_retirement_unhappy_deny2")
        else:
            dispatcher.utter_message(
                response="utter_retirement_unhappy_general_guidance"
            )
            result.append(FollowupAction("financial360_form"))
        return result


class ActionHaystack(Action):
    def name(self) -> Text:
        return "action_call_haystack"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        url = "http://localhost:8000/query"
        payload = {"query": str(tracker.latest_message["text"])}
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, json=payload).json()

        if response["answers"]:
            answer = response["answers"][0]["answer"]
        else:
            answer = "No Answer Found!"

        dispatcher.utter_message(text=answer)

        return []


class ActionLangChain(Action):
    def name(self) -> Text:
        return "action_langchain"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        # file_from_local = os.path.expanduser("~/Downloads/529_Plans.txt")
        # dataset = pd.read_table(
        #     file_from_local, header=None, delimiter=None, skip_blank_lines=True
        # )[0].to_list()
        # dataset = load_dataset(
        #     "WealthBuild/Investing_Knowledgebase",
        #     use_auth_token=hf_token,
        # )["train"]["text"]
        # string_dataset = " ".join(dataset)

        # llm = HuggingFaceHub(
        #     repo_id="mosaicml/mpt-7b",
        #     model_kwargs={"temperature": 0.1},
        # )

        # chain = load_qa_chain(llm=llm, chain_type="map_reduce")
        query = tracker.latest_message["text"]
        # handler = StdOutCallbackHandler()

        # result = chain.run(
        #     input_documents=similar_docs, question=query, callbacks=[handler]
        # )
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retreiver
        )

        # )
        # chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")
        # qa = RetrievalQAWithSourcesChain.from_chain_type(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=db.as_retriever(search_kwargs={"k": 5}),
        #     return_source_documents=True,
        # )
        # result = qa.run(query)
        result = qa.run(query)
        # answer = result["result"]
        # source_docs = result["source_documents"]
        # result = qa({"query": query})
        # answer = result["result"]
        # sources = result["sources"]
        # documents = result["source_documents"]

        # vector_dimension = data_embeddings.shape[1]
        # index = faiss.IndexFlatL2(vector_dimension)
        # faiss.normalize_L2(data_embeddings)
        # index.add(data_embeddings)

        # search_text = tracker.latest_message["text"]
        # search_embeddings = model.encode(search_text)
        # search_vector = np.array([search_embeddings])
        # faiss.normalize_L2(search_vector)

        # k = 5
        # d, i = index.search(search_vector, k=k)
        # print(i)
        dispatcher.utter_message(text=result)
        return []


class ActionGeneralLangChain(Action):
    def name(self) -> Text:
        return "action_general_langchain"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        user_message = tracker.latest_message["text"]
        latest_bot_message = tracker.get_last_event_for(event_type="bot")["text"]
        previous_action_name = tracker.get_last_event_for(
            event_type="action", exclude=["action_listen"]
        )["name"]
        latest_intent = tracker.get_intent_of_latest_message(skip_fallback_intent=False)
        print(
            f"latest intent = {latest_intent}, previous action name = {previous_action_name}"
        )

        if (
            previous_action_name == "action_general_langchain"
            and latest_intent == "affirm"
        ):
            result = retrieval_qa_chain_with_affirm(latest_bot_message)

        elif previous_action_name == "action_general_langchain" and (
            latest_intent == "knowledge_question" or latest_intent == "nlu_fallback"
        ):
            message = (
                f"Bot Message: {latest_bot_message} \n User Message: {user_message}"
            )
            result = retrieval_qa_chain_with_intent(message)

        elif (
            latest_intent == "knowledge_question" or latest_intent == "nlu_fallback"
        ) and previous_action_name != "action_general_langchain":
            result = llm_chain(user_message)

        dispatcher.utter_message(text=result)
        return []


# class ActionInsertName(Action):
#     def name(self) -> Text:
#         return "action_insert_name"

#     def run(
#         self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
#     ) -> List[EventType]:

#         first_name = tracker.get_slot("first_name")
#         last_name = tracker.get_slot("last_name")
#         cursor.execute("""INSERT INTO Person (first_name, last_name) VALUES (?,?)""",
#                     f"{first_name}", f"{last_name}").rowcount
#         cnxn.commit()
#         return []
