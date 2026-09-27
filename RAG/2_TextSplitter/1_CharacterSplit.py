# from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap=0,
#     separator=''
# )
# text = """Create a single **16:9 professional Smart India Hackathon (SIH) presentation slide** titled **“Feasibility & Viability”** for the project:

# **“Integrated Polar Science Outreach, Knowledge Repository and Media Dissemination Portal”**

# Design a clean, modern, technology-focused slide suitable for an SIH jury presentation. Use a **polar science visual theme** with subtle Arctic/Antarctic elements such as ice, snow, polar research, satellite data, and scientific data visualization. Keep the design professional and avoid excessive decoration.

# Structure the slide into **4 clearly separated sections/cards**:

# ### 1. Technical Feasibility

# * Cloud-based and scalable architecture
# * Modular web portal with responsive UI
# * Centralized scientific knowledge repository
# * APIs for integrating datasets, publications, media and external scientific resources
# * AI/NLP support for intelligent search, summarization and knowledge discovery

# ### 2. Operational Feasibility

# * Simple interface for researchers, students, educators and the public
# * Role-based access for content management
# * Easy content upload, verification and publishing workflow
# * Search, filtering and categorized knowledge access
# * Designed for both technical and non-technical users

# ### 3. Financial & Scalability Viability

# * Uses open-source technologies wherever possible
# * Cloud deployment allows resources to scale based on usage
# * Modular architecture reduces future development and maintenance costs
# * Can support increasing users, datasets, publications and multimedia content
# * Sustainable for long-term institutional deployment

# ### 4. Long-Term Impact & Sustainability

# * Creates a centralized digital ecosystem for polar science outreach
# * Improves accessibility and discoverability of polar research
# * Supports education, awareness and scientific collaboration
# * Enables continuous addition of new datasets, research and media
# * Can evolve into a national-level polar science knowledge platform

# At the bottom, add a concise highlighted statement:

# **“Technically achievable • Cost-effective • Scalable • Sustainable • High societal and scientific impact”**

# Use a **blue, white and subtle cyan color palette**, clean typography, minimal text, modern icons, balanced spacing and strong visual hierarchy. Use small relevant icons for technology, operations, scalability and impact. Make the slide readable from a projector and suitable for a **2–3 minute SIH pitch**.

# Do not overcrowd the slide. Keep each section concise and use visual elements such as a small scalability/cloud diagram or impact flow where appropriate.
# """
# res = splitter.split_text(text)

# print(res)


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('PDF\Syanopsis CCTV Survailance.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap=0
)

res = splitter.split_documents(docs)

print(res)
