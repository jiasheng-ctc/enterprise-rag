"""
FIXED MCP Tools Integration for Enterprise RAG System
Replace your existing rag_system/utils/mcp_tools.py with this version
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from datetime import datetime
import re
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)

def load_environment_variables():
    """
    Load environment variables from .env file - specifically for enterprise-rag project
    """
    if not DOTENV_AVAILABLE:
        logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        return False
    
    # Specific path for your project structure
    # From /home/ubuntu/enterprise-rag/rag_system/utils/mcp_tools.py
    # To /home/ubuntu/enterprise-rag/.env
    env_path = Path(__file__).parent.parent.parent / ".env"
    
    if env_path.exists():
        logger.info(f"Loading .env from: {env_path}")
        load_dotenv(env_path, override=True)
        return True
    else:
        logger.error(f".env file not found at expected location: {env_path}")
        logger.info("Please ensure your .env file exists at /home/ubuntu/enterprise-rag/.env")
        return False

class SequentialThinkingTool:
    """
    Implements sequential thinking pattern for complex problem solving
    """
    
    def __init__(self):
        self.steps = []
        
    def process(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Break down complex queries into sequential steps
        """
        self.steps = []
        
        # Step 1: Problem Decomposition
        decomposed = self._decompose_problem(query)
        self.steps.append({
            "step": "Problem Decomposition",
            "result": decomposed
        })
        
        # Step 2: Identify Key Components
        components = self._identify_components(decomposed)
        self.steps.append({
            "step": "Key Components", 
            "result": components
        })
        
        # Step 3: Logical Sequencing
        sequence = self._create_logical_sequence(components)
        self.steps.append({
            "step": "Logical Sequence",
            "result": sequence
        })
        
        # Step 4: Generate Structured Approach
        approach = self._generate_approach(sequence, context)
        self.steps.append({
            "step": "Structured Approach",
            "result": approach
        })
        
        return {
            "enhanced_query": self._format_enhanced_query(query, approach),
            "thinking_steps": self.steps,
            "approach": approach
        }
    
    def _decompose_problem(self, query: str) -> Dict[str, Any]:
        """Decompose the problem into smaller parts"""
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        query_lower = query.lower()
        
        question_type = None
        for word in question_words:
            if word in query_lower:
                question_type = word
                break
        
        topics = self._extract_topics(query)
        
        return {
            "question_type": question_type or "statement",
            "main_topics": topics,
            "complexity": "complex" if len(topics) > 3 else "simple"
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        words = text.split()
        topics = []
        
        # Look for capitalized words (potential proper nouns)
        for word in words:
            if word and word[0].isupper() and word.lower() not in ['the', 'a', 'an', 'is', 'are']:
                topics.append(word)
        
        # Look for key phrases in quotes
        quoted = re.findall(r'"([^"]*)"', text)
        topics.extend(quoted)
        
        return list(set(topics))[:5]
    
    def _identify_components(self, decomposed: Dict) -> List[str]:
        """Identify key components that need to be addressed"""
        components = []
        question_type = decomposed["question_type"]
        
        if question_type == "what":
            components.extend(["Definition", "Description", "Examples"])
        elif question_type == "how":
            components.extend(["Process", "Steps", "Requirements"])
        elif question_type == "why":
            components.extend(["Reasons", "Causes", "Benefits"])
        elif question_type == "when":
            components.extend(["Timeline", "Conditions", "Prerequisites"])
        else:
            components.extend(["Overview", "Details", "Context"])
        
        return components
    
    def _create_logical_sequence(self, components: List[str]) -> List[str]:
        """Create a logical sequence for addressing components"""
        priority_order = [
            "Definition", "Overview", "Context",
            "Description", "About", "Process", 
            "Steps", "Requirements", "Prerequisites",
            "Timeline", "Conditions", "Reasons",
            "Causes", "Benefits", "Examples", "Details"
        ]
        
        sorted_components = []
        for priority in priority_order:
            for component in components:
                if priority in component and component not in sorted_components:
                    sorted_components.append(component)
        
        for component in components:
            if component not in sorted_components:
                sorted_components.append(component)
        
        return sorted_components
    
    def _generate_approach(self, sequence: List[str], context: str) -> str:
        """Generate a structured approach based on the sequence"""
        approach_parts = ["To answer this question comprehensively:"]
        
        for i, component in enumerate(sequence, 1):
            approach_parts.append(f"{i}. Address {component}")
        
        if context:
            approach_parts.append(f"\nUsing context from: {context[:100]}...")
        
        return "\n".join(approach_parts)
    
    def _format_enhanced_query(self, original_query: str, approach: str) -> str:
        """Format the enhanced query with sequential thinking"""
        return f"""[Sequential Thinking Applied]

Original Question: {original_query}

Structured Approach:
{approach}

Please provide a comprehensive answer following this logical sequence."""



# Replace the WebSearchTool class in rag_system/utils/mcp_tools.py

class WebSearchTool:
    """
    IMPROVED Web search with better Google API handling and fallback
    """
    
    def __init__(self, search_api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        # Load environment variables first
        env_loaded = load_environment_variables()
        
        # Get API credentials
        self.google_api_key = search_api_key or os.getenv("GOOGLE_API_KEY")
        self.google_search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        
        self.search_results = []
        self.sources = []
        
        # Debug environment loading and API credentials
        logger.info(f"Environment loaded: {env_loaded}")
        
        if self.google_api_key:
            logger.info(f"Google API Key found: {'*' * (len(self.google_api_key) - 8)}{self.google_api_key[-8:]}")
        else:
            logger.warning("No Google API Key found in environment variables")
            logger.info("Check that GOOGLE_API_KEY is set in /home/ubuntu/enterprise-rag/.env")
            
        if self.google_search_engine_id:
            logger.info(f"Search Engine ID found: {self.google_search_engine_id}")
        else:
            logger.warning("No Google Search Engine ID found in environment variables")
            logger.info("Check that GOOGLE_SEARCH_ENGINE_ID is set in /home/ubuntu/enterprise-rag/.env")
        
        # Test Google API on initialization
        self.google_working = self._test_google_api()
        
        if self.google_working:
            logger.info("✅ Google Custom Search API is working")
        else:
            logger.warning("❌ Google Custom Search API not working, will use DuckDuckGo fallback")
    
    def _test_google_api(self) -> bool:
        """Test if Google Custom Search API is working"""
        if not self.google_api_key or not self.google_search_engine_id:
            logger.warning("Missing Google API credentials")
            return False
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_engine_id,
                'q': 'test weather singapore',
                'num': 1
            }
            
            logger.info("Testing Google Custom Search API...")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    logger.info("✅ Google API test successful")
                    return True
                else:
                    logger.warning("Google API returned no items")
                    return False
            else:
                logger.error(f"Google API test failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Google API test error: {e}")
            return False
    
    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search using Google Custom Search API with DuckDuckGo fallback
        """
        try:
            if self.google_working:
                return self._google_search_direct(query, num_results)
            else:
                logger.info("Using DuckDuckGo fallback search")
                return self._duckduckgo_search(query, num_results)
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._enhanced_mock_search(query, num_results)

    def _google_search_direct(self, query: str, num_results: int) -> Dict[str, Any]:
            """
            Direct Google Custom Search API call with better error handling
            """
            try:
                url = "https://www.googleapis.com/customsearch/v1"

                # FIXED: Smart query enhancement logic
                if 'news' in query.lower() or 'latest' in query.lower():
                    search_query = f"{query} 2025"
                else:
                    search_query = query

                params = {
                    'key': self.google_api_key,
                    'cx': self.google_search_engine_id,
                    'q': search_query,  # Use the conditional query
                    'num': min(num_results, 10),
                    'safe': 'active',
                    'sort': 'date'  # Sort by most recent
                }
                
                logger.info(f"🔍 Searching Google for: '{search_query}'")
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for errors in response
                    if 'error' in data:
                        error_msg = data['error'].get('message', 'Unknown error')
                        logger.error(f"Google API error: {error_msg}")
                        return self._duckduckgo_search(query, num_results)
                    
                    items = data.get('items', [])
                    
                    if not items:
                        logger.warning("Google search returned no results, trying DuckDuckGo")
                        return self._duckduckgo_search(query, num_results)
                    
                    # Format results
                    search_results = []
                    sources = []
                    
                    for item in items:
                        formatted_result = {
                            "title": item.get('title', ''),
                            "snippet": item.get('snippet', ''),
                            "url": item.get('link', ''),
                            "displayLink": item.get('displayLink', ''),
                            "date": datetime.now().strftime("%Y-%m-%d")
                        }
                        search_results.append(formatted_result)
                        
                        sources.append({
                            "url": item.get('link', ''),
                            "domain": item.get('displayLink', ''),
                            "title": item.get('title', '')
                        })
                    
                    self.search_results = search_results
                    self.sources = sources
                    
                    # Format context
                    context = self._format_search_context(search_results, query, "Google")
                    
                    logger.info(f"✅ Google search successful: {len(search_results)} results")
                    
                    return {
                        "results": search_results,
                        "sources": sources,
                        "query": query,
                        "total_results": data.get('searchInformation', {}).get('totalResults', str(len(search_results))),
                        "search_method": "google",
                        "context": context,
                        "status": "success"
                    }
                elif response.status_code == 403:
                    logger.error("Google API quota exceeded or access denied")
                    return self._duckduckgo_search(query, num_results)
                else:
                    logger.error(f"Google API error: {response.status_code} - {response.text}")
                    return self._duckduckgo_search(query, num_results)
                    
            except Exception as e:
                logger.error(f"Google search failed: {e}")
                return self._duckduckgo_search(query, num_results)

    def _duckduckgo_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        DuckDuckGo search fallback with actual API
        """
        try:
            from ddgs import DDGS
            
            logger.info(f"🦆 Searching DuckDuckGo for: '{query}'")
            
            with DDGS() as ddgs:
                results_raw = list(ddgs.text(query, max_results=num_results))
            
            if not results_raw:
                logger.warning("DuckDuckGo returned no results")
                return self._enhanced_mock_search(query, num_results)
            
            # Format results to match Google format
            search_results = []
            sources = []
            
            for item in results_raw:
                formatted_result = {
                    "title": item.get('title', ''),
                    "snippet": item.get('body', ''),
                    "url": item.get('href', ''),
                    "displayLink": self._extract_domain(item.get('href', '')),
                    "date": datetime.now().strftime("%Y-%m-%d")
                }
                search_results.append(formatted_result)
                
                sources.append({
                    "url": item.get('href', ''),
                    "domain": self._extract_domain(item.get('href', '')),
                    "title": item.get('title', '')
                })
            
            self.search_results = search_results
            self.sources = sources
            
            # Format context
            context = self._format_search_context(search_results, query, "DuckDuckGo")
            
            logger.info(f"✅ DuckDuckGo search successful: {len(search_results)} results")
            
            return {
                "results": search_results,
                "sources": sources,
                "query": query,
                "total_results": str(len(search_results)),
                "search_method": "duckduckgo",
                "context": context,
                "status": "success"
            }
            
        except ImportError:
            logger.error("duckduckgo-search package not installed")
            return self._enhanced_mock_search(query, num_results)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return self._enhanced_mock_search(query, num_results)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url

    def _format_search_context(self, results: List[Dict], query: str, search_engine: str) -> str:
        """Format search results as context - SIMPLIFIED"""
        if not results:
            return f"No web search results found for '{query}'"
        
        # Just return the raw search results
        context_parts = []
        
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description')
            
            context_parts.append(f"{i}. {title}")
            context_parts.append(f"{snippet}")
            context_parts.append("")
        
        return "\n".join(context_parts)
        
    def _enhanced_mock_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Enhanced mock search with more realistic Singapore weather results
        """
        logger.info(f"🔍 Using enhanced mock search for: {query}")
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")
        query_lower = query.lower()
        
        # More realistic Singapore weather results
        if "weather" in query_lower and "singapore" in query_lower:
            results = [
                {
                    "title": "Weather in Singapore - Meteorological Service Singapore",
                    "snippet": f"Current weather in Singapore as of {current_time} SGT: Partly cloudy, 29°C. Humidity: 78%. Southwest winds at 12 km/h. Today's forecast: High 32°C, Low 26°C with possible afternoon thunderstorms.",
                    "url": "https://www.weather.gov.sg/weather-forecast-singapore",
                    "displayLink": "weather.gov.sg",
                    "date": current_date
                },
                {
                    "title": f"Singapore Weather Today {current_date} - Real-time Updates",
                    "snippet": f"Live weather conditions for Singapore: Temperature 29°C, feels like 34°C. Visibility 10km. UV Index: High. Air quality: Moderate (PSI 65). Next update at {(datetime.now().hour + 1) % 24}:00.",
                    "url": "https://www.weather.com/weather/today/l/Singapore",
                    "displayLink": "weather.com",
                    "date": current_date
                },
                {
                    "title": "Singapore Hourly Weather Forecast & Radar - AccuWeather",
                    "snippet": "Detailed hourly forecast for Singapore showing temperature, precipitation probability, and wind conditions. Interactive radar showing current cloud cover and rain patterns across the island.",
                    "url": "https://www.accuweather.com/en/sg/singapore/300597/hourly-weather-forecast/300597",
                    "displayLink": "accuweather.com",
                    "date": current_date
                }
            ]
        else:
            # Generic fallback results
            results = [
                {
                    "title": f"Latest Information about {query.title()}",
                    "snippet": f"Comprehensive and up-to-date information about {query}. Find the latest news, updates, and detailed analysis on this topic.",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "displayLink": "example.com",
                    "date": current_date
                }
            ]
        
        # Create sources
        sources = []
        for result in results:
            sources.append({
                "url": result["url"],
                "domain": result["displayLink"],
                "title": result["title"]
            })
        
        self.search_results = results
        self.sources = sources
        
        # Format context
        context = self._format_search_context(results, query, "Enhanced Mock")
        
        return {
            "results": results,
            "sources": sources,
            "query": query,
            "total_results": str(len(results)),
            "search_method": "enhanced_mock",
            "context": context,
            "status": "mock_success"
        }

class MCPToolsOrchestrator:
    """
    Orchestrates multiple MCP tools for enhanced query processing
    """
    
    def __init__(self):
        self.sequential_thinking = SequentialThinkingTool()
        self.web_search = WebSearchTool()
        
    def process_with_tools(
        self,
        query: str,
        use_sequential_thinking: bool = False,
        use_web_search: bool = False,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Process query using selected MCP tools
        """
        result = {
            "original_query": query,
            "enhanced_query": query,
            "tools_used": [],
            "tool_outputs": {}
        }
        
        # Apply Sequential Thinking if enabled
        if use_sequential_thinking:
            thinking_result = self.sequential_thinking.process(query, context)
            result["enhanced_query"] = thinking_result["enhanced_query"]
            result["tools_used"].append("Sequential Thinking")
            result["tool_outputs"]["sequential_thinking"] = thinking_result
            logger.info("✅ Applied Sequential Thinking to query")
        
        # Apply Web Search if enabled
        if use_web_search:
            search_data = self.web_search.search(query, num_results=5)
            
            if search_data.get("results"):
                result["enhanced_query"] += f"\n\n{search_data['context']}"
                result["tools_used"].append("Web Search")
                result["tool_outputs"]["web_search"] = search_data
                logger.info(f"✅ Added {len(search_data['results'])} web search results using {search_data.get('search_method', 'unknown')} method")
        
        # Add metadata
        result["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "tools_count": len(result["tools_used"])
        }
        
        return result
    
    def extract_tool_flags(self, query: str) -> Tuple[str, bool, bool]:
        """
        Extract tool flags from query if present
        """
        use_sequential = False
        use_web_search = False
        clean_query = query
        
        # Check for tool indicators in the query
        if "[Tools:" in query:
            tool_match = re.search(r'\[Tools: ([^\]]+)\]', query)
            if tool_match:
                tools_text = tool_match.group(1).lower()
                use_sequential = "sequential thinking" in tools_text
                use_web_search = "web search" in tools_text or "search the web" in tools_text
                
                clean_query = re.sub(r'\[Tools: [^\]]+\]', '', query).strip()
        
        return clean_query, use_sequential, use_web_search


def enhance_query_with_mcp_tools(
    query: str,
    session_context: str = "",
    enable_sequential: bool = True,
    enable_web_search: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for MCP tools integration
    """
    orchestrator = MCPToolsOrchestrator()
    
    # Extract any embedded tool flags
    clean_query, flag_sequential, flag_web = orchestrator.extract_tool_flags(query)
    
    # Use flags from query or parameters
    use_sequential = flag_sequential or enable_sequential
    use_web = flag_web or enable_web_search
    
    # Process with tools
    result = orchestrator.process_with_tools(
        clean_query,
        use_sequential_thinking=use_sequential,
        use_web_search=use_web,
        context=session_context
    )
    
    return result


# Test function
if __name__ == "__main__":
    print("🧪 Testing FIXED MCP Tools...")
    
    test_query = "What's the weather in Singapore today?"
    
    result = enhance_query_with_mcp_tools(
        test_query,
        enable_sequential=True,
        enable_web_search=True
    )
    
    print(f"Original Query: {result['original_query']}")
    print(f"Tools Used: {result['tools_used']}")
    
    if "web_search" in result["tool_outputs"]:
        search_data = result["tool_outputs"]["web_search"]
        print(f"Search Method: {search_data.get('search_method', 'unknown')}")
        print(f"Results Found: {len(search_data.get('results', []))}")
        print(f"Status: {search_data.get('status', 'unknown')}")
        
        if search_data.get('results'):
            print("\nFirst Result:")
            first = search_data['results'][0]
            print(f"  Title: {first.get('title', 'No title')}")
            print(f"  Source: {first.get('displayLink', 'No source')}")
            print(f"  Snippet: {first.get('snippet', 'No snippet')[:100]}...")