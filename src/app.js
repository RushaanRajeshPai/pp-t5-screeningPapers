const express = require('express');
const { StateGraph, END } = require('@langchain/langgraph');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { HumanMessage, SystemMessage } = require('@langchain/core/messages');

const app = express();
app.use(express.json({ limit: '50mb' }));
require('dotenv').config();

// Initialize Gemini model
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.3,
  apiKey: process.env.GOOGLE_API_KEY
});

// State schema for the workflow
class WorkflowState {
  constructor() {
    this.inputPapers = [];
    this.extractedMetadata = [];
    this.generatedCriteria = [];
    this.evaluationResults = [];
    this.criteriaStats = {};
    this.finalSelectedPapers = [];
    this.currentStep = '';
    this.errors = [];
  }
}

// Agent 1: Input Processing
async function agent1_processInput(state) {
  console.log("🤖 Agent 1: Processing input papers...");
  
  try {
    // Validate input
    if (!state.inputPapers || state.inputPapers.length !== 50) {
      throw new Error("Expected exactly 50 research papers");
    }
    
    // Validate each paper has required fields
    for (let i = 0; i < state.inputPapers.length; i++) {
      const paper = state.inputPapers[i];
      if (!paper.title || !paper.abstract) {
        throw new Error(`Paper ${i + 1} missing required fields: title, or abstract`);
      }
    }
    
    state.currentStep = 'Input Processed';
    console.log(`✅ Agent 1: Successfully validated ${state.inputPapers.length} papers`);
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 1 Error: ${error.message}`);
    throw error;
  }
}

// Agent 2: Metadata Extraction
async function agent2_extractMetadata(state) {
  console.log("🤖 Agent 2: Extracting metadata from papers...");
  
  try {
    const extractedMetadata = [];
    
    for (let i = 0; i < state.inputPapers.length; i++) {
      const paper = state.inputPapers[i];
      
      const prompt = `
      Extract comprehensive metadata from this research paper. Return a JSON object with the following structure:
      {
        "title": "paper title",
        "authors": ["author1", "author2"],
        "journal": "journal name",
        "year": 2024,
        "keywords": ["keyword1", "keyword2"],
        "research_domain": "primary research field",
        "methodology": "research methodology used",
        "sample_size": "if applicable",
        "study_type": "experimental/observational/review/etc",
        "main_findings": "brief summary of key findings",
        "limitations": "study limitations if mentioned",
        "abstract_summary": "concise abstract summary"
      }
      
      Paper Details:
      Title: ${paper.title}
      Abstract: ${paper.abstract}
      
      Extract only factual information present in the paper. If information is not available, use "Not specified".
      `;
      
      const response = await model.invoke([
        new SystemMessage("You are a research paper metadata extraction expert. Extract accurate metadata and return valid JSON only."),
        new HumanMessage(prompt)
      ]);
      
      try {
        const metadata = JSON.parse(response.content);
        metadata.paper_id = i + 1;
        metadata.original_index = i;
        extractedMetadata.push(metadata);
      } catch (parseError) {
        // Fallback extraction if JSON parsing fails
        extractedMetadata.push({
          paper_id: i + 1,
          original_index: i,
          title: paper.title,
          abstract_summary: paper.abstract.substring(0, 200),
          research_domain: "Not specified",
          methodology: "Not specified",
          study_type: "Not specified",
          main_findings: "Not specified",
          keywords: [],
          authors: [],
          journal: "Not specified",
          year: "Not specified"
        });
      }
      
      // Progress indicator
      if ((i + 1) % 10 === 0) {
        console.log(`📊 Agent 2: Processed ${i + 1}/50 papers`);
      }
    }
    
    state.extractedMetadata = extractedMetadata;
    state.currentStep = 'Metadata Extracted';
    console.log("✅ Agent 2: Metadata extraction completed");
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 2 Error: ${error.message}`);
    throw error;
  }
}

// Agent 3: Generate Screening Criteria
async function agent3_generateCriteria(state) {
  console.log("🤖 Agent 3: Generating screening criteria...");
  
  try {
    // Analyze metadata to understand the research landscape
    const metadataSummary = state.extractedMetadata.map(paper => ({
      title: paper.title,
      research_domain: paper.research_domain,
      methodology: paper.methodology,
      study_type: paper.study_type,
      keywords: paper.keywords
    }));
    
    const prompt = `
    Based on the metadata of 50 research papers, generate 6 comprehensive screening criteria/questions that would help identify the most relevant and high-quality papers for systematic review.
    
    Metadata Summary:
    ${JSON.stringify(metadataSummary, null, 2)}
    
    Generate 6 criteria that can be answered with Yes/Maybe/No. Each criterion should:
    1. Be specific and measurable
    2. Focus on different aspects (methodology, relevance, quality, scope, etc.)
    3. Help distinguish between high-quality and lower-quality papers
    4. Be applicable to the research domain represented in these papers
    
    Return a JSON array with this structure:
    [
      {
        "id": 1,
        "criterion": "Clear research question statement",
        "description": "Does the paper clearly state its research question or hypothesis?",
        "evaluation_focus": "clarity and specificity of research objectives"
      },
      // ... 5 more criteria
    ]
    
    Make sure criteria are relevant to the research domain and can effectively screen papers.
    `;
    
    const response = await model.invoke([
      new SystemMessage("You are a systematic review expert. Generate comprehensive screening criteria that will effectively filter research papers for quality and relevance."),
      new HumanMessage(prompt)
    ]);
    
    const criteria = JSON.parse(response.content);
    
    if (!Array.isArray(criteria) || criteria.length !== 6) {
      throw new Error("Failed to generate exactly 6 criteria");
    }
    
    state.generatedCriteria = criteria;
    state.currentStep = 'Criteria Generated';
    console.log("✅ Agent 3: Generated 6 screening criteria");
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 3 Error: ${error.message}`);
    throw error;
  }
}

// Agent 4: Evaluate Papers Against Criteria
async function agent4_evaluatePapers(state) {
  console.log("🤖 Agent 4: Evaluating papers against criteria...");
  
  try {
    const evaluationResults = [];
    
    for (let i = 0; i < state.inputPapers.length; i++) {
      const paper = state.inputPapers[i];
      const metadata = state.extractedMetadata[i];
      
      const criteriaText = state.generatedCriteria.map(c => 
        `Criterion ${c.id}: ${c.criterion} - ${c.description}`
      ).join('\n');
      
      const prompt = `
      Evaluate this research paper against the following 6 criteria. For each criterion, respond with exactly "Yes", "Maybe", or "No" based on the paper metadata.
      
      CRITERIA:
      ${criteriaText}
      
      PAPER TO EVALUATE:
      Title: ${paper.title}
      Abstract: ${paper.abstract}
      
      EVALUATION GUIDELINES:
      - "Yes": Paper clearly meets the criterion
      - "Maybe": Paper partially meets the criterion or unclear evidence
      - "No": Paper does not meet the criterion
      
      Return ONLY a JSON object in this exact format:
      {
        "paper_id": ${i + 1},
        "title": "${paper.title}",
        "evaluations": [
          {"criterion_id": 1, "response": "Yes/Maybe/No", "reasoning": "brief explanation"},
          {"criterion_id": 2, "response": "Yes/Maybe/No", "reasoning": "brief explanation"},
          {"criterion_id": 3, "response": "Yes/Maybe/No", "reasoning": "brief explanation"},
          {"criterion_id": 4, "response": "Yes/Maybe/No", "reasoning": "brief explanation"},
          {"criterion_id": 5, "response": "Yes/Maybe/No", "reasoning": "brief explanation"},
          {"criterion_id": 6, "response": "Yes/Maybe/No", "reasoning": "brief explanation"}
        ]
      }
      `;
      
      const response = await model.invoke([
        new SystemMessage("You are a systematic review expert. Evaluate research papers objectively against screening criteria. Return only valid JSON."),
        new HumanMessage(prompt)
      ]);
      
      try {
        const evaluation = JSON.parse(response.content);
        
        // Validate evaluation structure
        if (!evaluation.evaluations || evaluation.evaluations.length !== 6) {
          throw new Error("Invalid evaluation structure");
        }
        
        // Ensure all responses are valid
        evaluation.evaluations = evaluation.evaluations.map(eval => ({
          ...eval,
          response: ['Yes', 'Maybe', 'No'].includes(eval.response) ? eval.response : 'No'
        }));
        
        evaluationResults.push(evaluation);
      } catch (parseError) {
        // Fallback evaluation
        evaluationResults.push({
          paper_id: i + 1,
          title: paper.title,
          evaluations: state.generatedCriteria.map(c => ({
            criterion_id: c.id,
            response: "Maybe",
            reasoning: "Evaluation failed, marked as Maybe"
          }))
        });
      }
      
      // Progress indicator
      if ((i + 1) % 10 === 0) {
        console.log(`📊 Agent 4: Evaluated ${i + 1}/50 papers`);
      }
    }
    
    state.evaluationResults = evaluationResults;
    state.currentStep = 'Papers Evaluated';
    console.log("✅ Agent 4: Paper evaluation completed");
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 4 Error: ${error.message}`);
    throw error;
  }
}

// Agent 5: Generate Statistics
async function agent5_generateStats(state) {
  console.log("🤖 Agent 5: Generating criteria statistics...");
  
  try {
    const stats = {};
    
    // Initialize stats for each criterion
    state.generatedCriteria.forEach(criterion => {
      stats[criterion.id] = {
        criterion: criterion.criterion,
        description: criterion.description,
        yes_count: 0,
        maybe_count: 0,
        no_count: 0,
        yes_papers: [],
        maybe_papers: [],
        no_papers: []
      };
    });
    
    // Count responses for each criterion
    state.evaluationResults.forEach(paperEval => {
      paperEval.evaluations.forEach(eval => {
        const criterionStat = stats[eval.criterion_id];
        if (criterionStat) {
          switch (eval.response) {
            case 'Yes':
              criterionStat.yes_count++;
              criterionStat.yes_papers.push({
                paper_id: paperEval.paper_id,
                title: paperEval.title,
                reasoning: eval.reasoning
              });
              break;
            case 'Maybe':
              criterionStat.maybe_count++;
              criterionStat.maybe_papers.push({
                paper_id: paperEval.paper_id,
                title: paperEval.title,
                reasoning: eval.reasoning
              });
              break;
            case 'No':
              criterionStat.no_count++;
              criterionStat.no_papers.push({
                paper_id: paperEval.paper_id,
                title: paperEval.title,
                reasoning: eval.reasoning
              });
              break;
          }
        }
      });
    });
    
    state.criteriaStats = stats;
    state.currentStep = 'Statistics Generated';
    console.log("✅ Agent 5: Statistics generation completed");
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 5 Error: ${error.message}`);
    throw error;
  }
}

// Agent 6: Select Top 10 Papers
async function agent6_selectTopPapers(state) {
  console.log("🤖 Agent 6: Selecting top 10 papers...");
  
  try {
    const scoredPapers = [];
    
    // Score each paper based on evaluation results
    state.evaluationResults.forEach(paperEval => {
      let yesCount = 0;
      let maybeCount = 0;
      let noCount = 0;
      
      paperEval.evaluations.forEach(eval => {
        switch (eval.response) {
          case 'Yes': yesCount++; break;
          case 'Maybe': maybeCount++; break;
          case 'No': noCount++; break;
        }
      });
      
      // Calculate eligibility score
      // Priority: All Yes > 5 Yes + 1 Maybe > 4 Yes + 2 Maybe, etc.
      let eligibilityScore = 0;
      let isEligible = false;
      
      if (yesCount === 6) {
        eligibilityScore = 1000; // Highest priority
        isEligible = true;
      } else if (yesCount === 5 && maybeCount === 1) {
        eligibilityScore = 900;
        isEligible = true;
      } else if (yesCount === 4 && maybeCount === 2) {
        eligibilityScore = 800;
        isEligible = true;
      } else if (yesCount >= 3 && (yesCount + maybeCount) >= 5) {
        eligibilityScore = 700;
        isEligible = true;
      }
      
      // Add bonus points for quality indicators
      eligibilityScore += (yesCount * 10) + (maybeCount * 5);
      
      scoredPapers.push({
        paper_id: paperEval.paper_id,
        title: paperEval.title,
        yes_count: yesCount,
        maybe_count: maybeCount,
        no_count: noCount,
        eligibility_score: eligibilityScore,
        is_eligible: isEligible,
        evaluations: paperEval.evaluations,
        original_index: paperEval.paper_id - 1
      });
    });
    
    // Sort by eligibility score (descending)
    scoredPapers.sort((a, b) => b.eligibility_score - a.eligibility_score);
    
    // Select top 10 eligible papers, or top 10 overall if not enough eligible
    const eligiblePapers = scoredPapers.filter(p => p.is_eligible);
    let selectedPapers;
    
    if (eligiblePapers.length >= 10) {
      selectedPapers = eligiblePapers.slice(0, 10);
    } else {
      // If less than 10 eligible, take all eligible + highest scoring non-eligible
      const nonEligible = scoredPapers.filter(p => !p.is_eligible);
      const needed = 10 - eligiblePapers.length;
      selectedPapers = [...eligiblePapers, ...nonEligible.slice(0, needed)];
    }
    
    // Add original paper data
    selectedPapers = selectedPapers.map(paper => ({
      ...paper,
      original_paper: state.inputPapers[paper.original_index],
      metadata: state.extractedMetadata[paper.original_index]
    }));
    
    state.finalSelectedPapers = selectedPapers;
    state.currentStep = 'Top 10 Selected';
    console.log(`✅ Agent 6: Selected top ${selectedPapers.length} papers`);
    
    return state;
  } catch (error) {
    state.errors.push(`Agent 6 Error: ${error.message}`);
    throw error;
  }
}

// Create the workflow graph
function createWorkflow() {
  const workflow = new StateGraph(WorkflowState);
  
  // Add nodes (agents)
  workflow.addNode("agent1", agent1_processInput);
  workflow.addNode("agent2", agent2_extractMetadata);
  workflow.addNode("agent3", agent3_generateCriteria);
  workflow.addNode("agent4", agent4_evaluatePapers);
  workflow.addNode("agent5", agent5_generateStats);
  workflow.addNode("agent6", agent6_selectTopPapers);
  
  // Define the flow
  workflow.setEntryPoint("agent1");
  workflow.addEdge("agent1", "agent2");
  workflow.addEdge("agent2", "agent3");
  workflow.addEdge("agent3", "agent4");
  workflow.addEdge("agent4", "agent5");
  workflow.addEdge("agent5", "agent6");
  workflow.addEdge("agent6", END);
  
  return workflow.compile();
}

// API Routes
app.post('/screen-papers', async (req, res) => {
  try {
    console.log("🚀 Starting research paper screening workflow...");
    
    const { papers } = req.body;
    
    if (!papers || !Array.isArray(papers)) {
      return res.status(400).json({
        error: "Invalid input: 'papers' should be an array",
        required_format: {
          papers: [
            {
              title: "Paper title",
              abstract: "Paper abstract"
            }
          ]
        }
      });
    }
    
    // Initialize state
    const initialState = new WorkflowState();
    initialState.inputPapers = papers;
    
    // Create and run workflow
    const workflow = createWorkflow();
    const finalState = await workflow.invoke(initialState);
    
    // Prepare response
    const response = {
      success: true,
      workflow_steps: finalState.currentStep,
      input_papers_count: finalState.inputPapers.length,
      generated_criteria: finalState.generatedCriteria,
      criteria_statistics: finalState.criteriaStats,
      selected_papers_count: finalState.finalSelectedPapers.length,
      selected_papers: finalState.finalSelectedPapers.map(paper => ({
        rank: finalState.finalSelectedPapers.indexOf(paper) + 1,
        paper_id: paper.paper_id,
        title: paper.title,
        eligibility_score: paper.eligibility_score,
        criteria_results: {
          yes_count: paper.yes_count,
          maybe_count: paper.maybe_count,
          no_count: paper.no_count
        },
        detailed_evaluations: paper.evaluations,
        metadata: paper.metadata
      })),
      errors: finalState.errors
    };
    
    console.log("✅ Workflow completed successfully!");
    res.json(response);
    
  } catch (error) {
    console.error("❌ Workflow failed:", error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'Research Paper Screening Agent',
    timestamp: new Date().toISOString()
  });
});

// Get workflow status
app.get('/workflow-info', (req, res) => {
  res.json({
    service: "Research Paper Screening Agentic Workflow",
    agents: [
      { id: 1, name: "Input Processor", function: "Validate and process input papers" },
      { id: 2, name: "Metadata Extractor", function: "Extract comprehensive metadata from papers" },
      { id: 3, name: "Criteria Generator", function: "Generate 6 screening criteria based on metadata" },
      { id: 4, name: "Paper Evaluator", function: "Evaluate each paper against criteria (Yes/Maybe/No)" },
      { id: 5, name: "Statistics Generator", function: "Generate statistics for criteria responses" },
      { id: 6, name: "Top Papers Selector", function: "Select top 10 papers based on evaluation scores" }
    ],
    model: "gemini-2.0-flash-exp",
    framework: "LangGraph + LangChain"
  });
});

// Example endpoint for testing with dummy data
app.get('/test-dummy', async (req, res) => {
  const dummyPapers = Array.from({ length: 50 }, (_, i) => ({
    title: `Research Paper ${i + 1}: Impact of AI on Healthcare Systems`,
    abstract: `This study examines the implementation of artificial intelligence in healthcare systems. The research focuses on efficiency improvements, cost reduction, and patient outcome enhancement. We conducted a comprehensive analysis of ${Math.floor(Math.random() * 1000) + 100} healthcare facilities over a ${Math.floor(Math.random() * 3) + 1}-year period.`,
  }));
  
  try {
    const workflow = createWorkflow();
    const initialState = new WorkflowState();
    initialState.inputPapers = dummyPapers;
    
    const finalState = await workflow.invoke(initialState);
    res.json({ message: "Test completed successfully", results: finalState });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Research Paper Screening Agent running on port ${PORT}`);
  console.log(`📊 Workflow Info: http://localhost:${PORT}/workflow-info`);
  console.log(`🔍 Screen Papers: POST http://localhost:${PORT}/screen-papers`);
  console.log(`🧪 Test Dummy: GET http://localhost:${PORT}/test-dummy`);
});

module.exports = app;