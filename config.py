import os
from dotenv import load_dotenv
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings



load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

POLICY_METADATA = {

    "Remote_Work_Policy.pdf": {
        "source": "Remote_Work_Policy.pdf",
        "department": "HR",
        "policy_type": "Remote Work",
        "policy_id": "HR-RW-001",
        "version": "2.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Leave_Policy.pdf": {
        "source": "Leave_Policy.pdf",
        "department": "HR",
        "policy_type": "Leave",
        "policy_id": "HR-LV-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "IT_Policy.pdf": {
        "source": "IT_Policy.pdf",
        "department": "IT",
        "policy_type": "IT",
        "policy_id": "IT-IT-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "International_Work_Policy.pdf": {
        "source": "International_Work_Policy.pdf",
        "department": "HR",
        "policy_type": "International Work",
        "policy_id": "HR-IW-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },


    "Asset_Management_Policy.pdf": {
        "source": "Asset_Management_Policy.pdf",
        "department": "HR",
        "policy_type": "Asset Management",
        "policy_id": "HR-AM-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Travel_Expense_Policy.pdf": {
        "source": "Travel_Expense_Policy.pdf",
        "department": "Finance",
        "policy_type": "Travel & Expense",
        "policy_id": "FIN-TE-001",
        "version": "2.1",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Attendance_Work_From_Office_Policy.pdf": {
        "source": "Attendance_Work_From_Office_Policy.pdf",
        "department": "HR",
        "policy_type": "Attendance Work From Office",
        "policy_id": "HR-AW-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Data_Privacy_Protection_Policy.pdf": {
        "source": "Data_Privacy_Protection_Policy.pdf",
        "department": "HR",
        "policy_type": "Data Privacy Protection",
        "policy_id": "HR-DP-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Employee_Offboarding_Separation_Policy.pdf": {
        "source": "Employee_Offboarding_Separation_Policy.pdf",
        "department": "HR",
        "policy_type": "Employee Offboarding Separation",
        "policy_id": "HR-EO-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Employment Policy and Practices.pdf": {
        "source": "Employment Policy and Practices.pdf",
        "department": "HR",
        "policy_type": "Employment Policy and Practices",
        "policy_id": "HR-EP-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Information_Classification_Data_Handling_Policy.pdf": {
        "source": "Information_Classification_Data_Handling_Policy.pdf",
        "department": "HR",
        "policy_type": "Information Classification Data Handling",
        "policy_id": "HR-IC-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    },

    "Information_Security_Acceptable_Use_Policy.pdf": {
        "source": "Information_Security_Acceptable_Use_Policy.pdf",
        "department": "HR",
        "policy_type": "Information Security Acceptable Use",
        "policy_id": "HR-IS-001",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "status": "active"
    }

    }