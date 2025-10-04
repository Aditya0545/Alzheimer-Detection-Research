import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { Readable } from 'stream'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Validate required fields
    const requiredFields = [
      'age', 'gender', 'ethnicity', 'education', 'bmi',
      'smoking', 'alcohol', 'physical_activity', 'diet_quality', 'sleep_quality',
      'family_history', 'cardiovascular', 'diabetes', 'depression', 'head_injury',
      'hypertension', 'systolic_bp', 'diastolic_bp', 'cholesterol_total', 'cholesterol_ldl',
      'cholesterol_hdl', 'cholesterol_triglycerides', 'mmse', 'functional_assessment',
      'memory_complaints', 'behavioral_problems', 'adl', 'confusion',
      'disorientation', 'personality_changes', 'difficulty_tasks'
    ]
    
    for (const field of requiredFields) {
      if (!(field in body)) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        )
      }
    }
    
    // For Vercel deployment, we'll return mock data since Python won't be available
    // In a production environment, you would need to use a different approach
    // such as serverless functions or external API
    
    // Return mock prediction data for demonstration
    const mockResult = {
      voting_prediction: Math.random() > 0.5 ? 1 : 0,
      stacking_prediction: Math.random() > 0.5 ? 1 : 0,
      voting_probabilities: {
        low_risk: Math.random(),
        high_risk: Math.random()
      },
      stacking_probabilities: {
        low_risk: Math.random(),
        high_risk: Math.random()
      },
      models_agree: Math.random() > 0.5,
      feature_count: 39
    }
    
    return NextResponse.json(mockResult)
    
  } catch (error) {
    console.error('API Error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}