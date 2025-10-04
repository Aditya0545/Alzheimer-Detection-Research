'use client'

import { useState, useEffect } from 'react'
import { Brain, Shield, AlertTriangle, CheckCircle, BarChart3, Users, Activity, Heart, Target, TrendingUp, Award, Zap, FileText, BookOpen, ExternalLink } from 'lucide-react'

interface PredictionResult {
  voting_prediction: number
  stacking_prediction: number
  voting_probabilities: {
    low_risk: number
    high_risk: number
  }
  stacking_probabilities: {
    low_risk: number
    high_risk: number
  }
  models_agree: boolean
  feature_count: number
}

export default function Home() {
  const [showLanding, setShowLanding] = useState(true)
  const [selectedView, setSelectedView] = useState<'prediction' | 'analysis'>('analysis')
  const [selectedModel, setSelectedModel] = useState<'svm' | 'decision_tree' | 'random_forest' | 'knn' | 'comparison'>('comparison')
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [documentContent, setDocumentContent] = useState<string>('')
  const [loadingDocument, setLoadingDocument] = useState(false)
  const [formData, setFormData] = useState({
    age: 60,
    gender: 'Male',
    ethnicity: 'Caucasian',
    education: 'Bachelor\'s',
    bmi: 22.0,
    smoking: 'No',
    alcohol: 2.0,
    physical_activity: 5.0,
    diet_quality: 6.0,
    sleep_quality: 5.0,
    family_history: 'No',
    cardiovascular: 'No',
    diabetes: 'No',
    depression: 'No',
    head_injury: 'No',
    hypertension: 'No',
    systolic_bp: 120,
    diastolic_bp: 80,
    cholesterol_total: 200.0,
    cholesterol_ldl: 120.0,
    cholesterol_hdl: 50.0,
    cholesterol_triglycerides: 150.0,
    mmse: 25,
    functional_assessment: 7.0,
    memory_complaints: 'No',
    behavioral_problems: 'No',
    adl: 8.0,
    confusion: 'No',
    disorientation: 'No',
    personality_changes: 'No',
    difficulty_tasks: 'No'
  })

  // Model performance data from research
  const modelPerformance = {
    random_forest: {
      name: 'Random Forest',
      accuracy: 92.3,
      precision: 93.0,
      recall: 92.3,
      f1_score: 92.2,
      icon: Target,
      color: 'green-600',
      description: 'Best overall performance with ensemble of decision trees'
    },
    decision_tree: {
      name: 'Decision Tree',
      accuracy: 89.3,
      precision: 89.2,
      recall: 89.3,
      f1_score: 89.2,
      icon: BarChart3,
      color: 'blue-600',
      description: 'Interpretable model with clear decision paths'
    },
    svm: {
      name: 'Support Vector Machine',
      accuracy: 83.3,
      precision: 83.1,
      recall: 83.3,
      f1_score: 83.0,
      icon: Shield,
      color: 'purple-600',
      description: 'Robust classifier with RBF kernel'
    },
    knn: {
      name: 'K-Nearest Neighbors',
      accuracy: 72.3,
      precision: 71.4,
      recall: 72.3,
      f1_score: 71.0,
      icon: Users,
      color: 'orange-600',
      description: 'Instance-based learning with k=5 neighbors'
    }
  }

  // Document mapping for each model
  const modelDocuments = {
    comparison: [
      { name: 'Comprehensive Research Report', file: 'comprehensive-research-report.md', description: 'Complete analysis of all models' },
      { name: 'Model Comparison Analysis', file: 'model-comparison-analysis.md', description: 'Detailed performance comparison' },
      { name: 'Ensemble Methods Analysis', file: 'ensemble-methods-analysis.md', description: 'Voting and stacking ensemble analysis' },
      { name: 'Hyperparameter Tuning Guide', file: 'hyperparameter-tuning-analysis.md', description: 'Optimization strategies for all models' },
      { name: 'Implementation Guide', file: 'implementation-guide.md', description: 'Complete deployment guide' }
    ],
    random_forest: [
      { name: 'Random Forest Analysis', file: 'random-forest-analysis.md', description: 'Detailed analysis of best performing model (92.3%)' },
      { name: 'Ensemble Methods Analysis', file: 'ensemble-methods-analysis.md', description: 'How Random Forest fits in ensemble approach' },
      { name: 'Hyperparameter Tuning Guide', file: 'hyperparameter-tuning-analysis.md', description: 'Optimization strategies for Random Forest' }
    ],
    decision_tree: [
      { name: 'Decision Tree Analysis', file: 'decision-tree-analysis.md', description: 'Interpretable model analysis (89.3% accuracy)' },
      { name: 'Hyperparameter Tuning Guide', file: 'hyperparameter-tuning-analysis.md', description: 'Optimization strategies for Decision Tree' }
    ],
    svm: [
      { name: 'SVM Analysis', file: 'svm-analysis.md', description: 'Support Vector Machine analysis (83.3% accuracy)' },
      { name: 'Hyperparameter Tuning Guide', file: 'hyperparameter-tuning-analysis.md', description: 'Optimization strategies for SVM' }
    ],
    knn: [
      { name: 'KNN Analysis', file: 'knn-analysis.md', description: 'K-Nearest Neighbors analysis (72.3% accuracy)' },
      { name: 'Hyperparameter Tuning Guide', file: 'hyperparameter-tuning-analysis.md', description: 'Optimization strategies for KNN' }
    ]
  }

  // Function to load document content
  const loadDocument = async (filename: string) => {
    setLoadingDocument(true)
    try {
      const response = await fetch(`/docs/${filename}`)
      if (response.ok) {
        const content = await response.text()
        setDocumentContent(content)
        setSelectedDocument(filename)
      } else {
        setDocumentContent('Document not found or could not be loaded.')
        setSelectedDocument(filename)
      }
    } catch (error) {
      setDocumentContent('Error loading document. Please try again.')
      setSelectedDocument(filename)
    }
    setLoadingDocument(false)
  }

  // Function to render markdown content with proper formatting
  const renderMarkdown = (content: string) => {
    const lines = content.split('\n')
    const renderedElements: JSX.Element[] = []
    let inCodeBlock = false
    let codeBlockContent: string[] = []
    let inTable = false
    let tableRows: string[][] = []
    let isTableHeader = false
    let inList = false
    let listItems: string[] = []
    let listType = 'ul' // 'ul' or 'ol'
    
    const processLine = (line: string, index: number) => {
      // Handle code blocks
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          // End code block
          renderedElements.push(
            <div key={`code-${index}`} className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-sm my-4 overflow-x-auto">
              <pre>{codeBlockContent.join('\n')}</pre>
            </div>
          )
          codeBlockContent = []
          inCodeBlock = false
        } else {
          // Start code block
          inCodeBlock = true
        }
        return
      }
      
      if (inCodeBlock) {
        codeBlockContent.push(line)
        return
      }
      
      // Handle tables
      if (line.includes('|') && line.trim().length > 0) {
        if (!inTable) {
          inTable = true
          isTableHeader = true
        }
        
        const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell.length > 0)
        if (cells.length > 0) {
          tableRows.push(cells)
        }
        
        // Check if next line is table separator
        const nextLine = lines[index + 1]
        if (nextLine && nextLine.includes('---')) {
          return // Skip separator line
        }
        
        // Check if this is the last row of table
        const nextLineAfter = lines[index + 1]
        if (!nextLineAfter || !nextLineAfter.includes('|')) {
          // End of table, render it
          renderedElements.push(
            <div key={`table-${index}`} className="overflow-x-auto my-6">
              <table className="min-w-full bg-white border border-gray-300 rounded-lg shadow-sm">
                <thead className="bg-gradient-to-r from-blue-50 to-purple-50">
                  <tr>
                    {tableRows[0].map((header, i) => (
                      <th key={i} className="px-4 py-3 text-left text-sm font-semibold text-gray-900 border-b border-gray-300">
                        {formatInlineMarkdown(header)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.slice(1).map((row, rowIndex) => (
                    <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex} className="px-4 py-3 text-sm text-gray-700 border-b border-gray-200">
                          {formatInlineMarkdown(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
          tableRows = []
          inTable = false
          isTableHeader = false
        }
        return
      }
      
      // Handle lists
      if (line.match(/^\s*[-*+]\s/) || line.match(/^\s*\d+\.\s/)) {
        const isOrdered = line.match(/^\s*\d+\.\s/)
        const content = line.replace(/^\s*[-*+\d+\.]\s/, '')
        
        if (!inList) {
          inList = true
          listType = isOrdered ? 'ol' : 'ul'
        }
        
        listItems.push(content)
        
        // Check if next line is also a list item
        const nextLine = lines[index + 1]
        if (!nextLine || (!nextLine.match(/^\s*[-*+]\s/) && !nextLine.match(/^\s*\d+\.\s/))) {
          // End of list, render it
          const ListComponent = listType === 'ol' ? 'ol' : 'ul'
          renderedElements.push(
            <ListComponent key={`list-${index}`} className={`my-4 ml-6 space-y-2 ${
              listType === 'ol' ? 'list-decimal' : 'list-disc'
            }`}>
              {listItems.map((item, i) => (
                <li key={i} className="text-gray-700 leading-relaxed">
                  {formatInlineMarkdown(item)}
                </li>
              ))}
            </ListComponent>
          )
          listItems = []
          inList = false
        }
        return
      }
      
      // Handle headings
      if (line.startsWith('# ')) {
        renderedElements.push(
          <h1 key={index} className="text-4xl font-bold text-gray-900 mb-6 mt-8 border-b-2 border-blue-200 pb-2">
            {formatInlineMarkdown(line.replace('# ', ''))}
          </h1>
        )
        return
      }
      
      if (line.startsWith('## ')) {
        renderedElements.push(
          <h2 key={index} className="text-3xl font-semibold text-gray-800 mb-5 mt-7 border-b border-gray-300 pb-2">
            {formatInlineMarkdown(line.replace('## ', ''))}
          </h2>
        )
        return
      }
      
      if (line.startsWith('### ')) {
        renderedElements.push(
          <h3 key={index} className="text-2xl font-semibold text-gray-700 mb-4 mt-6">
            {formatInlineMarkdown(line.replace('### ', ''))}
          </h3>
        )
        return
      }
      
      if (line.startsWith('#### ')) {
        renderedElements.push(
          <h4 key={index} className="text-xl font-medium text-gray-600 mb-3 mt-5">
            {formatInlineMarkdown(line.replace('#### ', ''))}
          </h4>
        )
        return
      }
      
      if (line.startsWith('##### ')) {
        renderedElements.push(
          <h5 key={index} className="text-lg font-medium text-gray-600 mb-2 mt-4">
            {formatInlineMarkdown(line.replace('##### ', ''))}
          </h5>
        )
        return
      }
      
      // Handle empty lines
      if (line.trim() === '') {
        renderedElements.push(<div key={index} className="h-3" />)
        return
      }
      
      // Handle horizontal rule
      if (line.trim() === '---') {
        renderedElements.push(
          <hr key={index} className="my-8 border-gray-300" />
        )
        return
      }
      
      // Handle blockquotes
      if (line.startsWith('> ')) {
        renderedElements.push(
          <blockquote key={index} className="border-l-4 border-blue-500 pl-4 py-2 my-4 bg-blue-50 text-gray-700 italic">
            {formatInlineMarkdown(line.replace('> ', ''))}
          </blockquote>
        )
        return
      }
      
      // Handle regular paragraphs
      if (line.trim().length > 0) {
        renderedElements.push(
          <p key={index} className="text-gray-700 leading-relaxed my-3">
            {formatInlineMarkdown(line)}
          </p>
        )
      }
    }
    
    lines.forEach(processLine)
    
    return <div className="prose prose-lg max-w-none">{renderedElements}</div>
  }
  
  // Function to format inline markdown (bold, italic, code, links)
  const formatInlineMarkdown = (text: string): React.ReactNode => {
    if (!text) return text
    
    // Simple approach: process each type of formatting sequentially
    const parts: React.ReactNode[] = []
    let keyCounter = 0
    
    // Split by bold text **text** first
    const boldRegex = /\*\*([^*]+)\*\*/g
    const boldSplit = text.split(boldRegex)
    
    for (let i = 0; i < boldSplit.length; i++) {
      if (i % 2 === 0) {
        // Regular text - process for other formatting
        const segment = boldSplit[i]
        if (segment) {
          parts.push(...processNonBoldText(segment, keyCounter))
          keyCounter += 10 // Leave space for nested elements
        }
      } else {
        // Bold text
        parts.push(
          <strong key={`bold-${keyCounter++}`} className="font-bold text-gray-900">
            {boldSplit[i]}
          </strong>
        )
      }
    }
    
    return parts.length > 1 ? <>{parts}</> : (parts[0] || text)
  }
  
  // Helper function to process text that doesn't contain bold formatting
  const processNonBoldText = (text: string, startKey: number): React.ReactNode[] => {
    const parts: React.ReactNode[] = []
    let keyCounter = startKey
    
    // Process inline code `code` first
    const codeRegex = /`([^`]+)`/g
    const codeSplit = text.split(codeRegex)
    
    for (let i = 0; i < codeSplit.length; i++) {
      if (i % 2 === 0) {
        // Regular text - process for italic and links
        const segment = codeSplit[i]
        if (segment) {
          parts.push(...processTextForItalicAndLinks(segment, keyCounter))
          keyCounter += 5
        }
      } else {
        // Code text
        parts.push(
          <code key={`code-${keyCounter++}`} className="bg-gray-100 text-red-600 px-2 py-1 rounded font-mono text-sm">
            {codeSplit[i]}
          </code>
        )
      }
    }
    
    return parts
  }
  
  // Helper function to process text for italic and links
  const processTextForItalicAndLinks = (text: string, startKey: number): React.ReactNode[] => {
    const parts: React.ReactNode[] = []
    let keyCounter = startKey
    
    // Process links [text](url) first
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g
    const linkSplit = text.split(linkRegex)
    
    for (let i = 0; i < linkSplit.length; i += 3) {
      // Regular text
      if (linkSplit[i]) {
        parts.push(...processTextForItalic(linkSplit[i], keyCounter))
        keyCounter += 2
      }
      
      // Link (if exists)
      if (linkSplit[i + 1] && linkSplit[i + 2]) {
        parts.push(
          <a key={`link-${keyCounter++}`} href={linkSplit[i + 2]} target="_blank" rel="noopener noreferrer" 
             className="text-blue-600 hover:text-blue-800 underline">
            {linkSplit[i + 1]}
          </a>
        )
      }
    }
    
    return parts
  }
  
  // Helper function to process text for italic only
  const processTextForItalic = (text: string, startKey: number): React.ReactNode[] => {
    const parts: React.ReactNode[] = []
    let keyCounter = startKey
    
    // Process italic *text* (single asterisk, not double)
    const italicRegex = /(?<!\*)\*([^*]+)\*(?!\*)/g
    
    // Since lookbehind isn't supported everywhere, use a different approach
    let remaining = text
    let lastIndex = 0
    
    // Manual parsing to avoid ** being treated as italic
    while (remaining.length > 0) {
      const asteriskIndex = remaining.indexOf('*')
      if (asteriskIndex === -1) {
        // No more asterisks, add remaining text
        if (remaining) {
          parts.push(<span key={`text-${keyCounter++}`}>{remaining}</span>)
        }
        break
      }
      
      // Check if it's a double asterisk (bold) - skip it
      if (remaining[asteriskIndex + 1] === '*') {
        // Add text up to and including the **
        const beforeBold = remaining.substring(0, asteriskIndex + 2)
        if (beforeBold) {
          parts.push(<span key={`text-${keyCounter++}`}>{beforeBold}</span>)
        }
        remaining = remaining.substring(asteriskIndex + 2)
        continue
      }
      
      // Look for closing asterisk
      const closingIndex = remaining.indexOf('*', asteriskIndex + 1)
      if (closingIndex === -1) {
        // No closing asterisk, treat as regular text
        parts.push(<span key={`text-${keyCounter++}`}>{remaining}</span>)
        break
      }
      
      // Add text before italic
      if (asteriskIndex > 0) {
        parts.push(<span key={`text-${keyCounter++}`}>{remaining.substring(0, asteriskIndex)}</span>)
      }
      
      // Add italic text
      const italicText = remaining.substring(asteriskIndex + 1, closingIndex)
      parts.push(
        <em key={`italic-${keyCounter++}`} className="italic text-gray-800">
          {italicText}
        </em>
      )
      
      // Continue with remaining text
      remaining = remaining.substring(closingIndex + 1)
    }
    
    return parts.length > 0 ? parts : [<span key={`text-${startKey}`}>{text}</span>]
  }

  // Landing page component
  const renderLandingPage = () => {
    console.log('Rendering landing page, showLanding:', showLanding);
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="animate-fadeInUp">
            {/* Brain Icon with Animation */}
            <div className="mb-6 flex justify-center">
              <div className="relative">
                <Brain className="h-20 w-20 text-blue-600 animate-pulse-slow" />
                <div className="absolute inset-0 h-20 w-20 bg-blue-200 rounded-full opacity-20 animate-ping"></div>
              </div>
            </div>
            
            {/* Project Title */}
            <h1 className="text-4xl md:text-5xl font-bold gradient-text mb-4 animate-slideIn">
              Alzheimer's Disease Detection
            </h1>
            
            <h2 className="text-xl md:text-2xl font-semibold text-gray-700 mb-3">
              Machine Learning Research Project
            </h2>
            
            {/* Researcher Name */}
            <div className="mb-6 p-4 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-100">
              <p className="text-base text-gray-600 mb-1">Researcher</p>
              <h3 className="text-2xl font-bold text-gray-900">Er. P. Stanley</h3>
            </div>
            
            {/* Project Description */}
            <p className="text-lg text-gray-600 mb-6 max-w-3xl mx-auto leading-relaxed">
              A comprehensive comparative analysis of machine learning algorithms including 
              SVM, Decision Tree, Random Forest, and KNN for early Alzheimer's disease detection.
            </p>
            
            {/* Key Metrics Preview */}
            <div className="grid md:grid-cols-4 gap-3 mb-8">
              <div className="bg-white/60 backdrop-blur-sm rounded-xl p-3 border border-gray-100">
                <div className="text-xl font-bold text-green-600">92.3%</div>
                <div className="text-xs text-gray-600">Best Accuracy</div>
                <div className="text-xs text-gray-500">Random Forest</div>
              </div>
              <div className="bg-white/60 backdrop-blur-sm rounded-xl p-3 border border-gray-100">
                <div className="text-xl font-bold text-blue-600">4</div>
                <div className="text-xs text-gray-600">ML Models</div>
                <div className="text-xs text-gray-500">Compared</div>
              </div>
              <div className="bg-white/60 backdrop-blur-sm rounded-xl p-3 border border-gray-100">
                <div className="text-xl font-bold text-purple-600">2,149</div>
                <div className="text-xs text-gray-600">Patient Records</div>
                <div className="text-xs text-gray-500">Analyzed</div>
              </div>
              <div className="bg-white/60 backdrop-blur-sm rounded-xl p-3 border border-gray-100">
                <div className="text-xl font-bold text-orange-600">34</div>
                <div className="text-xs text-gray-600">Features</div>
                <div className="text-xs text-gray-500">Processed</div>
              </div>
            </div>
            
            {/* Enter Button - Fixed Implementation */}
            <div className="mt-8">
              <button 
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('Explore Research button clicked!');
                  setShowLanding(false);
                }}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold text-xl rounded-xl shadow-2xl hover:from-blue-700 hover:to-purple-700 hover:shadow-3xl transform hover:scale-105 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-300"
              >
                Explore Research →
              </button>
            </div>
            
            {/* Decorative Elements */}
            <div className="mt-8 flex justify-center space-x-6 opacity-60">
              <Target className="h-6 w-6 text-green-500 animate-bounce" style={{animationDelay: '0s'}} />
              <BarChart3 className="h-6 w-6 text-blue-500 animate-bounce" style={{animationDelay: '0.2s'}} />
              <Shield className="h-6 w-6 text-purple-500 animate-bounce" style={{animationDelay: '0.4s'}} />
              <Users className="h-6 w-6 text-orange-500 animate-bounce" style={{animationDelay: '0.6s'}} />
            </div>
          </div>
        </div>
      </div>
    )
  }

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Prediction failed')
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const renderModelSpecificAnalysis = (modelKey: string) => {
    switch (modelKey) {
      case 'random_forest':
        return (
          <div className="space-y-4">
            <p><strong>Algorithm:</strong> Ensemble of 100 decision trees with bootstrap sampling</p>
            <p><strong>Key Strengths:</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>Highest accuracy (92.3%) among all models</li>
              <li>Robust against overfitting due to ensemble nature</li>
              <li>Handles feature interactions well</li>
              <li>Provides feature importance rankings</li>
            </ul>
            <p><strong>Use Cases:</strong> Best choice for production deployment due to superior performance and stability</p>
          </div>
        )
      case 'decision_tree':
        return (
          <div className="space-y-4">
            <p><strong>Algorithm:</strong> Single decision tree with CART algorithm</p>
            <p><strong>Key Strengths:</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>High interpretability - easy to understand decision paths</li>
              <li>Strong performance (89.3% accuracy)</li>
              <li>Fast training and prediction</li>
              <li>Can handle both numerical and categorical features</li>
            </ul>
            <p><strong>Use Cases:</strong> Ideal when model interpretability is crucial for medical decision-making</p>
          </div>
        )
      case 'svm':
        return (
          <div className="space-y-4">
            <p><strong>Algorithm:</strong> Support Vector Machine with RBF kernel</p>
            <p><strong>Key Strengths:</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>Solid performance (83.3% accuracy)</li>
              <li>Effective in high-dimensional spaces</li>
              <li>Robust against outliers</li>
              <li>Memory efficient</li>
            </ul>
            <p><strong>Use Cases:</strong> Good baseline model with consistent performance across different datasets</p>
          </div>
        )
      case 'knn':
        return (
          <div className="space-y-4">
            <p><strong>Algorithm:</strong> K-Nearest Neighbors with k=5</p>
            <p><strong>Key Strengths:</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>Simple and intuitive algorithm</li>
              <li>No assumptions about data distribution</li>
              <li>Can capture local patterns in data</li>
              <li>Useful for similarity analysis</li>
            </ul>
            <p><strong>Limitations:</strong> Lower accuracy (72.3%) compared to other models, sensitive to irrelevant features</p>
          </div>
        )
      default:
        return null
    }
  }

  const renderModelAnalysis = () => {
    if (selectedDocument) {
      return (
        <div className="space-y-4 animate-fadeInUp"> {/* Compact layout with reduced spacing */}
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-3xl font-bold gradient-text">Research Documentation</h2>
            <button 
              onClick={() => {
                setSelectedDocument(null)
                setDocumentContent('')
              }}
              className="btn-secondary flex items-center gap-2 transform hover:scale-105 transition-all duration-300"
            >
              ← Back to {selectedModel === 'comparison' ? 'Comparison' : modelPerformance[selectedModel].name}
            </button>
          </div>

          <div className="bg-white rounded-xl shadow-2xl border-0 min-h-[80vh] max-h-[85vh] overflow-y-auto">
            {loadingDocument ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-3 text-gray-600 font-medium">Loading document...</span>
              </div>
            ) : (
              <div className="p-6"> {/* Adequate padding for readability */}
                {renderMarkdown(documentContent)}
              </div>
            )}
          </div>
        </div>
      )
    }
    if (selectedModel === 'comparison') {
      return (
        <div className="space-y-8">
          <h2 className="text-3xl font-bold text-gray-900">Model Performance Comparison</h2>
          
          {/* Performance Overview Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(modelPerformance).map(([key, model], index) => {
              const IconComponent = model.icon
              return (
                <div key={key} className="model-card animate-fadeInUp" style={{animationDelay: `${index * 0.1}s`}} onClick={() => setSelectedModel(key as any)}>
                  <div className="flex items-center mb-4">
                    <IconComponent className={`h-8 w-8 mr-3 ${
                      key === 'random_forest' ? 'text-green-600' :
                      key === 'decision_tree' ? 'text-blue-600' :
                      key === 'svm' ? 'text-purple-600' :
                      'text-orange-600'
                    }`} />
                    <h3 className="text-lg font-semibold">{model.name}</h3>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Accuracy</span>
                      <span className="font-semibold text-gray-900">{model.accuracy}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">F1 Score</span>
                      <span className="font-semibold text-gray-900">{model.f1_score}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                          key === 'random_forest' ? 'bg-gradient-to-r from-green-500 to-green-600' :
                          key === 'decision_tree' ? 'bg-gradient-to-r from-blue-500 to-blue-600' :
                          key === 'svm' ? 'bg-gradient-to-r from-purple-500 to-purple-600' :
                          'bg-gradient-to-r from-orange-500 to-orange-600'
                        }`} 
                        style={{width: `${model.accuracy}%`}}
                      ></div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Detailed Comparison Table */}
          <div className="card">
            <h3 className="text-xl font-semibold mb-6">Detailed Performance Metrics</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Model</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Accuracy</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Precision</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">Recall</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-900">F1 Score</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modelPerformance).map(([key, model]) => (
                    <tr key={key} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <model.icon className={`h-5 w-5 text-${model.color}-600 mr-2`} />
                          {model.name}
                        </div>
                      </td>
                      <td className="py-3 px-4 font-semibold">{model.accuracy}%</td>
                      <td className="py-3 px-4">{model.precision}%</td>
                      <td className="py-3 px-4">{model.recall}%</td>
                      <td className="py-3 px-4">{model.f1_score}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Insights */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Award className="h-6 w-6 text-yellow-600 mr-2" />
                Best Performer
              </h3>
              <div className="text-lg">
                <span className="font-bold text-green-600">Random Forest</span> achieved the highest accuracy at 
                <span className="font-bold"> 92.3%</span>, making it the most reliable model for Alzheimer's detection.
              </div>
            </div>
            
            <div className="card">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <TrendingUp className="h-6 w-6 text-blue-600 mr-2" />
                Performance Insights
              </h3>
              <ul className="space-y-2 text-sm">
                <li>• Random Forest: Best balance of accuracy and interpretability</li>
                <li>• Decision Tree: Most interpretable with 89.3% accuracy</li>
                <li>• SVM: Robust performance with 83.3% accuracy</li>
                <li>• KNN: Lower performance but useful for similarity analysis</li>
              </ul>
            </div>
          </div>

          {/* Research Documentation Section */}
          <div className="card">
            <h3 className="text-xl font-semibold mb-6 flex items-center">
              <BookOpen className="h-6 w-6 text-blue-600 mr-2" />
              Research Documentation
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {modelDocuments[selectedModel].map((doc, index) => (
                <div 
                  key={index}
                  onClick={() => loadDocument(doc.file)}
                  className="doc-card animate-slideIn"
                  style={{animationDelay: `${index * 0.1}s`}}
                >
                  <div className="flex items-start gap-3">
                    <FileText className="h-5 w-5 text-blue-600 mt-1 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-1">{doc.name}</h4>
                      <p className="text-sm text-gray-600 mb-2">{doc.description}</p>
                      <div className="flex items-center text-blue-600 text-sm font-medium">
                        <span>Read Documentation</span>
                        <ExternalLink className="h-3 w-3 ml-1" />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    }

    const model = modelPerformance[selectedModel]
    if (!model) return null

    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <model.icon className={`h-10 w-10 text-${model.color}-600 mr-4`} />
            <div>
              <h2 className="text-3xl font-bold text-gray-900">{model.name}</h2>
              <p className="text-gray-600">{model.description}</p>
            </div>
          </div>
          <button 
            onClick={() => setSelectedModel('comparison')}
            className="btn-secondary"
          >
            ← Back to Comparison
          </button>
        </div>

        {/* Model Performance Metrics */}
        <div className="grid md:grid-cols-4 gap-6">
          <div className="card text-center">
            <div className={`text-3xl font-bold text-${model.color}-600`}>{model.accuracy}%</div>
            <div className="text-sm text-gray-600 mt-1">Accuracy</div>
          </div>
          <div className="card text-center">
            <div className={`text-3xl font-bold text-${model.color}-600`}>{model.precision}%</div>
            <div className="text-sm text-gray-600 mt-1">Precision</div>
          </div>
          <div className="card text-center">
            <div className={`text-3xl font-bold text-${model.color}-600`}>{model.recall}%</div>
            <div className="text-sm text-gray-600 mt-1">Recall</div>
          </div>
          <div className="card text-center">
            <div className={`text-3xl font-bold text-${model.color}-600`}>{model.f1_score}%</div>
            <div className="text-sm text-gray-600 mt-1">F1 Score</div>
          </div>
        </div>

        {/* Model-specific Analysis */}
        <div className="card">
          <h3 className="text-xl font-semibold mb-4">Model Analysis</h3>
          {renderModelSpecificAnalysis(selectedModel)}
        </div>

        {/* Confusion Matrix Simulation */}
        <div className="card">
          <h3 className="text-xl font-semibold mb-4">Performance Visualization</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Confusion Matrix (Simulated)</h4>
              <div className="grid grid-cols-2 gap-2 w-48">
                <div className="bg-green-100 p-4 text-center rounded">
                  <div className="font-bold">TN</div>
                  <div className="text-sm">{Math.round(model.accuracy * 2)}</div>
                </div>
                <div className="bg-red-100 p-4 text-center rounded">
                  <div className="font-bold">FP</div>
                  <div className="text-sm">{Math.round((100 - model.accuracy) * 0.6)}</div>
                </div>
                <div className="bg-red-100 p-4 text-center rounded">
                  <div className="font-bold">FN</div>
                  <div className="text-sm">{Math.round((100 - model.accuracy) * 0.4)}</div>
                </div>
                <div className="bg-green-100 p-4 text-center rounded">
                  <div className="font-bold">TP</div>
                  <div className="text-sm">{Math.round(model.accuracy * 1.2)}</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Performance Chart</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Accuracy</span>
                    <span>{model.accuracy}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`bg-${model.color}-600 h-2 rounded-full`} style={{width: `${model.accuracy}%`}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Precision</span>
                    <span>{model.precision}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`bg-${model.color}-600 h-2 rounded-full`} style={{width: `${model.precision}%`}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Recall</span>
                    <span>{model.recall}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`bg-${model.color}-600 h-2 rounded-full`} style={{width: `${model.recall}%`}}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>F1 Score</span>
                    <span>{model.f1_score}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className={`bg-${model.color}-600 h-2 rounded-full`} style={{width: `${model.f1_score}%`}}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Documentation Section */}
        <div className="card">
          <h3 className="text-xl font-semibold mb-6 flex items-center">
            <BookOpen className="h-6 w-6 text-blue-600 mr-2" />
            {model.name} Documentation
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            {modelDocuments[selectedModel].map((doc, index) => (
              <div 
                key={index}
                onClick={() => loadDocument(doc.file)}
                className="doc-card animate-slideIn"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className="flex items-start gap-3">
                  <FileText className="h-5 w-5 text-blue-600 mt-1 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 mb-1">{doc.name}</h4>
                    <p className="text-sm text-gray-600 mb-2">{doc.description}</p>
                    <div className="flex items-center text-blue-600 text-sm font-medium">
                      <span>Read Documentation</span>
                      <ExternalLink className="h-3 w-3 ml-1" />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <>
      {showLanding ? (
        renderLandingPage()
      ) : (
        <div className="min-h-screen py-8 px-4">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="text-center mb-12 animate-fadeInUp">
              <div className="flex items-center justify-center mb-6 relative">
                {/* Back to Landing Button */}
                <button 
                  onClick={() => {
                    console.log('Back button clicked, setting showLanding to true')
                    setShowLanding(true)
                  }}
                  className="absolute left-0 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors duration-300 flex items-center gap-2 cursor-pointer bg-white rounded-lg shadow-sm"
                >
                  ← Back to Landing
                </button>
                
                <Brain className="h-12 w-12 text-blue-600 mr-4 animate-pulse-slow" />
                <h1 className="text-4xl font-bold gradient-text">Alzheimer's Detection Research</h1>
              </div>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Comprehensive analysis of machine learning models for Alzheimer's disease detection. 
                Explore individual model performance and ensemble methods.
              </p>
            </div>

            {/* Navigation */}
            <div className="flex justify-center mb-8">
              <div className="glass-effect p-2 rounded-xl shadow-lg">
                <button
                  onClick={() => setSelectedView('analysis')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 ${
                    selectedView === 'analysis' 
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                  }`}
                >
                  🔬 Model Analysis
                </button>
                <button
                  onClick={() => setSelectedView('prediction')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 ${
                    selectedView === 'prediction' 
                      ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                  }`}
                >
                  🩺 AI Prediction
                </button>
              </div>
            </div>

            {/* Content */}
            {selectedView === 'analysis' ? (
              renderModelAnalysis()
            ) : (
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Prediction form would go here - keeping existing form for now */}
                <div className="card">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Patient Information</h2>
                  <p className="text-gray-600">Use the ensemble prediction system to get AI-powered risk assessment.</p>
                  <button
                    onClick={() => setSelectedView('analysis')}
                    className="mt-4 btn-primary"
                  >
                    View Model Analysis Instead
                  </button>
                </div>
                
                <div className="card">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Ensemble Models</h2>
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h3 className="font-semibold text-blue-900">Voting Ensemble</h3>
                      <p className="text-blue-700 text-sm">91.86% accuracy using majority voting from 4 base models</p>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <h3 className="font-semibold text-green-900">Stacking Ensemble</h3>
                      <p className="text-green-700 text-sm">93.49% accuracy using meta-learner approach</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
