# Clario - AI-Powered 
           + Decision Intelligence Platform       
         2 +  
         3 +  ## 🚀 Overview
         4 +  
         5 +  Clario is a cutting-edge 
           + decision intelligence platform       
           +  that transforms how 
           + organizations make, track, and       
           +  analyze decisions. By 
           + combining advanced AI 
           + capabilities with 
           + sophisticated graph databases        
           + and vector search, Clario 
           + creates a living knowledge 
           + graph of your organization's         
           + decision-making patterns.
         6 +  
         7 +  ## 🎯 What We Do
         8 +  
         9 +  Clario helps organizations:
        10 +  - **Capture & Structure 
           + Decisions**: Automatically 
           + extract and structure 
           + decisions from meetings, 
           + documents, and communications        
        11 +  - **Build Knowledge Graphs**:       
           +  Create interconnected 
           + decision networks that reveal        
           + patterns, dependencies, and 
           + impacts
        12 +  - **Enable Intelligent 
           + Search**: Use semantic search        
           + to find relevant decisions, 
           + context, and historical 
           + patterns
        13 +  - **Provide AI Insights**: 
           + Deploy AI agents that analyze        
           + decision patterns and provide        
           + recommendations
        14 +  - **Track Decision Impact**:        
           + Monitor outcomes and learn 
           + from past decisions to improve       
           +  future choices
        15 +  
        16 +  ## 🏗️ Architecture
        17 +  
        18 +  ### Core Technologies
        19 +  
        20 +  - **Frontend**: React + 
           + TypeScript + Vite + 
           + TailwindCSS
        21 +  - **Backend**: FastAPI 
           + (Python)
        22 +  - **AI/LLM**: Friendli AI 
           + (Meta Llama 3.1)
        23 +  - **Vector Database**: 
           + Weaviate
        24 +  - **Graph Database**: Neo4j
        25 +  - **Voice Integration**: 
           + Vapi.ai
        26 +  - **Autonomous Agents**: 
           + Strands AI Framework
        27 +  
        28 +  ### Key Features
        29 +  
        30 +  #### 📊 Decision Graph Studio       
        31 +  - Interactive visualization         
           + of decision networks
        32 +  - Real-time graph exploration       
        33 +  - Pattern detection and 
           + analysis
        34 +  - Decision impact tracking
        35 +  
        36 +  #### 🤖 AI-Powered Analysis
        37 +  - Automated decision 
           + extraction from meetings
        38 +  - Semantic search across 
           + decision history
        39 +  - Pattern recognition and 
           + insights
        40 +  - Predictive recommendations        
        41 +  
        42 +  #### 🔍 Intelligent Search
        43 +  - Vector-based semantic 
           + search
        44 +  - Graph traversal queries
        45 +  - Context-aware results
        46 +  - Historical pattern matching       
        47 +  
        48 +  #### 🎙️ Voice Integration
        49 +  - Voice-activated decision 
           + capture
        50 +  - Real-time meeting 
           + transcription
        51 +  - Audio decision summaries
        52 +  - Voice-controlled navigation       
        53 +  
        54 +  #### 👥 Autonomous Agents
        55 +  - Decision analysis agents
        56 +  - Pattern detection agents
        57 +  - Recommendation agents
        58 +  - Monitoring and alerting 
           + agents
        59 +  
        60 +  ## 🛠️ Installation
        61 +  
        62 +  ### Prerequisites
        63 +  
        64 +  - Python 3.9+
        65 +  - Node.js 18+
        66 +  - Docker & Docker Compose 
           + (optional)
        67 +  
        68 +  ### Quick Start
        69 +  
        70 +  1. **Clone the repository**
        71 +  ```bash
        72 +  git clone https://github.com/       
           + xenn0010/Clario.git
        73 +  cd Clario
        74 +  ```
        75 +  
        76 +  2. **Set up environment 
           + variables**
        77 +  ```bash
        78 +  # Copy the example 
           + environment file
        79 +  cp env-templates/env.example        
           + backend/.env
        80 +  
        81 +  # Edit backend/.env with your       
           +  API keys:
        82 +  # - FRIENDLIAI_TOKEN 
           + (required)
        83 +  # - WEAVIATE_URL & 
           + WEAVIATE_API_KEY (required)
        84 +  # - NEO4J_URI, NEO4J_USER &         
           + NEO4J_PASSWORD (required)
        85 +  # - VAPI_API_KEY (optional 
           + for voice features)
        86 +  # - STRANDS_API_KEY (optional       
           +  for agents)
        87 +  ```
        88 +  
        89 +  3. **Install backend 
           + dependencies**
        90 +  ```bash
        91 +  cd backend
        92 +  python -m venv venv
        93 +  source venv/bin/activate  #         
           + On Windows: 
           + venv\Scripts\activate
        94 +  pip install -r 
           + requirements.txt
        95 +  ```
        96 +  
        97 +  4. **Install frontend 
           + dependencies**
        98 +  ```bash
        99 +  cd ../frontend
       100 +  npm install
       101 +  ```
       102 +  
       103 +  5. **Start the services**
       104 +  
       105 +  Backend:
       106 +  ```bash
       107 +  cd backend
       108 +  uvicorn main:app --reload 
           + --host 0.0.0.0 --port 8000
       109 +  ```
       110 +  
       111 +  Frontend:
       112 +  ```bash
       113 +  cd frontend
       114 +  npm run dev
       115 +  ```
       116 +  
       117 +  The application will be 
           + available at:
       118 +  - Frontend: 
           + http://localhost:5173
       119 +  - Backend API: 
           + http://localhost:8000
       120 +  - API Documentation: 
           + http://localhost:8000/docs
       121 +  
       122 +  ### Using Docker 
           + (Alternative)
       123 +  
       124 +  ```bash
       125 +  docker-compose up
       126 +  ```
       127 +  
       128 +  ## 🔑 API Keys & 
           + Configuration
       129 +  
       130 +  ### Required Services
       131 +  
       132 +  1. **Friendli AI** - For LLM        
           + capabilities
       133 +     - Get your API key at: 
           + https://friendli.ai
       134 +     - Model used: 
           + meta-llama-3.1-70b-instruct
       135 +  
       136 +  2. **Weaviate** - For vector        
           + search
       137 +     - Create a cluster at: 
           + https://console.weaviate.cloud       
       138 +     - Get your API key and URL       
       139 +  
       140 +  3. **Neo4j** - For graph 
           + database
       141 +     - Create an instance at:         
           + https://neo4j.com/cloud/aura/        
       142 +     - Note your connection 
           + details
       143 +  
       144 +  ### Optional Services
       145 +  
       146 +  - **Vapi.ai** - For voice 
           + interactions
       147 +  - **Strands AI** - For 
           + autonomous agents
       148 +  
       149 +  ## 📁 Project Structure
       150 +  
       151 +  ```
       152 +  Clario/
       153 +  ├── backend/
       154 +  │   ├── app/
       155 +  │   │   ├── api/           #        
           + API endpoints
       156 +  │   │   ├── core/          #        
           + Core configurations
       157 +  │   │   ├── models/        #        
           + Data models
       158 +  │   │   └── services/      #        
           + Business logic
       159 +  │   │       ├── ai/        #        
           + AI/LLM services
       160 +  │   │       ├── graph/     #        
           + Neo4j graph operations
       161 +  │   │       ├── vector/    #        
           + Weaviate vector operations
       162 +  │   │       ├── voice/     #        
           + Voice integration
       163 +  │   │       └── agents/    #        
           + Autonomous agents
       164 +  │   └── main.py           #         
           + FastAPI application
       165 +  │
       166 +  ├── frontend/
       167 +  │   ├── src/
       168 +  │   │   ├── components/    #        
           + React components
       169 +  │   │   ├── pages/        #         
           + Page components
       170 +  │   │   ├── layouts/      #         
           + Layout components
       171 +  │   │   └── types/        #         
           + TypeScript definitions
       172 +  │   └── package.json      #         
           + Frontend dependencies
       173 +  │
       174 +  └── docker-compose.yml    #         
           + Docker configuration
       175 +  ```
       176 +  
       177 +  ## 🧪 Testing the Integration       
       178 +  
       179 +  Run the Friendli AI test 
           + script to verify your setup:
       180 +  
       181 +  ```python
       182 +  python friendli_test.py
       183 +  ```
       184 +  
       185 +  ## 🚀 Usage
       186 +  
       187 +  ### Creating Your First 
           + Decision Graph
       188 +  
       189 +  1. **Start a Meeting**: Use         
           + the dashboard to create a new        
           + meeting
       190 +  2. **Capture Decisions**: The       
           +  AI will automatically extract       
           +  decisions from your input
       191 +  3. **Explore the Graph**: 
           + Navigate to Graph Studio to 
           + see your decision network
       192 +  4. **Search & Analyze**: Use        
           + the search feature to find 
           + related decisions
       193 +  5. **Get Insights**: Let AI         
           + agents analyze patterns and 
           + provide recommendations
       194 +  
       195 +  ### API Endpoints
       196 +  
       197 +  - `GET /api/v1/health` - 
           + Health check
       198 +  - `POST /api/v1/decisions` -        
           + Create a decision
       199 +  - `GET /api/v1/graph` - Get         
           + decision graph
       200 +  - `POST /api/v1/search` - 
           + Semantic search
       201 +  - `POST 
           + /api/v1/agents/analyze` - Run        
           + analysis agents
       202 +  
       203 +  Full API documentation 
           + available at: 
           + http://localhost:8000/docs
       204 +  
       205 +  ## 🤝 Contributing
       206 +  
       207 +  We welcome contributions! 
           + Please feel free to submit a         
           + Pull Request.
       208 +  
       209 +  ## 📄 License
       210 +  
       211 +  This project is licensed 
           + under the MIT License.
       212 +  
       213 +  ## 🆘 Support
       214 +  
       215 +  For issues, questions, or 
           + suggestions, please open an 
           + issue on GitHub.
       216 +  
       217 +  ## 🔮 Roadmap
       218 +  
       219 +  - [ ] Enhanced decision 
           + templates
       220 +  - [ ] Multi-language support        
       221 +  - [ ] Advanced analytics 
           + dashboard
       222 +  - [ ] Mobile application
       223 +  - [ ] Enterprise SSO 
           + integration
       224 +  - [ ] Custom AI agent builder       
       225 +  - [ ] Blockchain decision 
           + verification
       226 +  - [ ] Real-time collaboration       
           +  features
       227 +  
       228 +  ---
       229 +  
       230 +  Built with ❤️ using 
           + cutting-edge AI technologies         
           + to revolutionize 
           + organizational 
           + decision-making.
