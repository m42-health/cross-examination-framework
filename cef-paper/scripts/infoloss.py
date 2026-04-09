
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from openai import AsyncOpenAI
import logging
import yaml

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    overall_score: float
    factual_accuracy: float
    completeness: float
    semantic_preservation: float
    quality_level: str
    confidence_level: str
    explanation: str
    specific_issues: List[str]
    critical_errors: List[str]

class TranslationQualityAssessor:
    def __init__(self, 
                 base_url: str = "http://worker-1:8888/v1/",
                 api_key: str = "test",
                 default_model: str = "qwen2.5-72b-instruct",
                 prompt_filepath: Optional[Union[str, Path]] = None):
        """
        Initialize the translation quality assessor
        
        Args:
            base_url: URL address of the OpenAI-compatible API endpoint
            api_key: API key for the endpoint
            default_model: Default model name (e.g., "qwen2.5-72b-instruct")
            prompt_filepath: Optional path to prompts file
        """
        # Use AsyncOpenAI instead of LLM_Caller
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.default_model = default_model

    def create_assessment_messages(self, source_text: str, target_translation: str, target_language: str) -> List[Dict[str, str]]:
        """Create messages for the LLM to assess translation quality"""
        
        rubric = """
TRANSLATION QUALITY ASSESSMENT RUBRIC

Score Range: 0-5 with corresponding quality levels

SCORING CRITERIA:

**Score 5 - Excellent**
- All factual information accurately preserved
- All numerical data, dates, names, and technical terms correctly translated
- Complete semantic meaning maintained
- All entities, relationships, and context fully preserved
- No omissions or additions that change meaning
- Translation demonstrates exceptional accuracy and fidelity

**Score 4 - Good**
- Minor factual details may be slightly altered but core facts intact
- Occasional minor numerical or date discrepancies that don't affect main message
- Most entities and relationships preserved
- Minor semantic shifts that don't impact overall understanding
- 1-2 small omissions or additions that minimally affect meaning
- Translation maintains high quality with minor imperfections

**Score 3 - Acceptable**
- Some factual information lost or altered
- Notable errors in numbers, dates, or technical terms
- Some entities incorrectly translated or omitted
- Moderate semantic shifts affecting clarity
- Several omissions or additions that somewhat change meaning
- Core message still understandable but details compromised
- Translation meets basic standards but has noticeable issues

**Score 2 - Poor**
- Significant factual information lost or incorrectly translated
- Major errors in critical data (numbers, dates, names)
- Many entities missing or wrong
- Substantial semantic changes affecting comprehension
- Important concepts omitted or misrepresented
- Core message partially preserved but significantly altered
- Translation quality falls below acceptable standards

**Score 1 - Very Poor**
- Most factual information lost, altered, or incorrect
- Critical data severely corrupted or missing
- Entities and relationships largely incorrect
- Major semantic distortions
- Essential information omitted
- Core message barely recognizable
- Translation severely compromised

**Score 0 - Completely Inadequate**
- Complete loss of factual accuracy
- All or most critical information missing or wrong
- No recognizable preservation of original meaning
- Complete semantic breakdown
- Translation conveys different or opposite meaning
- Translation fails completely

EVALUATION DIMENSIONS:

1. **Factual Accuracy (40% weight)**
   - Correctness of facts, figures, dates, names
   - Preservation of quantitative information
   - Accuracy of technical/specialized terms

2. **Completeness (30% weight)**
   - No omission of important information
   - All key concepts included
   - No unauthorized additions

3. **Semantic Preservation (30% weight)**
   - Maintenance of original meaning
   - Preservation of relationships between concepts
   - Context and nuance retention
"""

        prompt = f"""
{rubric}

TASK: Assess the translation quality based on the rubric above.

SOURCE TEXT (English):
{source_text}

TARGET TRANSLATION ({target_language}):
{target_translation}

INSTRUCTIONS:
1. Carefully compare the source and target texts
2. Identify any factual discrepancies, omissions, or additions
3. Evaluate based on the rubric above
4. Provide scores for each dimension (0-5 scale)
5. Calculate overall weighted score
6. Determine quality level based on score
7. List specific issues found
8. Provide detailed explanation

OUTPUT FORMAT:
{{
  "overall_score": [0-5 with one decimal],
  "factual_accuracy": [0-5],
  "completeness": [0-5], 
  "semantic_preservation": [0-5],
  "quality_level": ["Excellent" | "Good" | "Acceptable" | "Poor" | "Very Poor" | "Completely Inadequate"],
  "confidence_level": ["high" | "medium" | "low"],
  "explanation": "Detailed explanation of assessment with justification for quality level",
  "specific_issues": [
    "Issue 1: Description with impact assessment",
    "Issue 2: Description with impact assessment"
  ],
  "preserved_elements": [
    "Element 1: What was preserved well",
    "Element 2: What was preserved well"  
  ],
  "critical_errors": [
    "Error 1: High-impact factual errors",
    "Error 2: Information completely lost"
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ]
}}

Provide your assessment:
"""
        
        system_message = {
            "role": "system",
            "content": "You are an expert translation quality assessor. Provide precise, objective assessments in the requested JSON format based on the quality levels defined in the rubric."
        }
        
        user_message = {
            "role": "user",
            "content": prompt
        }
        
        return [system_message, user_message]

    async def call_llm_api(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict:
        """Make async API call using AsyncOpenAI"""
        
        try:
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=2000,
                response_format={"type": "json_object"}  
            )
            
            # Convert response to dict format
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def parse_llm_response(self, response: Dict) -> QualityScore:
        """Parse LLM response and extract quality score"""
        
        try:
            # Extract content from response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Parse JSON response
            assessment = json.loads(content)
            
            return QualityScore(
                overall_score=float(assessment.get('overall_score', 0)),
                factual_accuracy=float(assessment.get('factual_accuracy', 0)),
                completeness=float(assessment.get('completeness', 0)),
                semantic_preservation=float(assessment.get('semantic_preservation', 0)),
                quality_level=assessment.get('quality_level', 'Completely Inadequate'),
                confidence_level=assessment.get('confidence_level', 'low'),
                explanation=assessment.get('explanation', ''),
                specific_issues=assessment.get('specific_issues', []),
                critical_errors=assessment.get('critical_errors', [])
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise Exception(f"Failed to parse LLM response: {str(e)}")

    async def assess_translation(self, source_text: str, target_translation: str, target_language: str, model: Optional[str] = None) -> QualityScore:
        """
        Main method to assess translation quality
        
        Args:
            source_text: Original English text
            target_translation: Translation in target language
            target_language: Target language name (e.g., 'Arabic', 'Japanese')
            model: Optional model override
            
        Returns:
            QualityScore object with detailed assessment
        """
        
        # Create assessment messages
        messages = self.create_assessment_messages(source_text, target_translation, target_language)
        
        # Call LLM API
        response = await self.call_llm_api(messages, model)
        
        # Parse and return results
        return self.parse_llm_response(response)

    async def batch_assess(self, translation_pairs: List[Tuple[str, str, str]], model: Optional[str] = None) -> List[QualityScore]:
        """
        Assess multiple translations in batch
        
        Args:
            translation_pairs: List of (source_text, target_translation, target_language) tuples
            model: Optional model override
            
        Returns:
            List of QualityScore objects
        """
        results = []
        for source, target, lang in translation_pairs:
            try:
                score = await self.assess_translation(source, target, lang, model)
                results.append(score)
            except Exception as e:
                logger.error(f"Error assessing translation: {str(e)}")
                # Add empty score for failed assessments
                results.append(QualityScore(0, 0, 0, 0, "Completely Inadequate", "low", f"Assessment failed: {str(e)}", [], []))
        
        return results


# Example usage
async def main():
    # Initialize assessor
    assessor = TranslationQualityAssessor(
        base_url="http://worker-13:8888/v1/",  
        api_key="test",  # Your API key
        default_model="/models_llm/Qwen2.5-72B-Instruct"  
    )
    
    # Example assessment
    source_text = """
    The company reported quarterly revenue of $2.5 billion, representing a 15% increase from the same period last year. 
    CEO John Smith announced that the new product launch scheduled for March 2024 has been delayed until June 2024 
    due to supply chain issues. The stock price rose 3.2% following the announcement.
    """
    
    # Example Arabic translation (intentionally with some errors for demonstration)
    arabic_translation = """
    أعلنت الشركة عن إيرادات ربع سنوية بقيمة 2.5 مليار دولار، مما يمثل زيادة بنسبة 10% عن نفس الفترة من العام الماضي.
    أعلن الرئيس التنفيذي جون سميث أن إطلاق المنتج الجديد المقرر في مارس 2024 قد تأجل حتى يوليو 2024
    بسبب مشاكل سلسلة التوريد. ارتفع سعر السهم بنسبة 3.2% بعد الإعلان.
    """
    
    try:
        # Assess translation quality
        score = await assessor.assess_translation(source_text, arabic_translation, "Arabic")
        
        # Print results
        print("=== TRANSLATION QUALITY ASSESSMENT ===")
        print(f"Overall Score: {score.overall_score}/5")
        print(f"Quality Level: {score.quality_level}")
        print(f"Confidence Level: {score.confidence_level}")
        print(f"\nDimensional Scores:")
        print(f"  Factual Accuracy: {score.factual_accuracy}/5")
        print(f"  Completeness: {score.completeness}/5") 
        print(f"  Semantic Preservation: {score.semantic_preservation}/5")
        print(f"\nExplanation:\n{score.explanation}")
        
        if score.critical_errors:
            print(f"\nCritical Errors:")
            for error in score.critical_errors:
                print(f"- {error}")
        
        print(f"\nSpecific Issues:")
        for issue in score.specific_issues:
            print(f"- {issue}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())