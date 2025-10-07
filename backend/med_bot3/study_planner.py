"""
Med‑Bot Study Planner
---------------------
A tool to generate semester study plans from a syllabus (PDF or TXT), export to PDF, and add events to Google/Outlook calendars using Arcade and Gemini.

Quick Start
-----------
1. Set your Gemini API key:
   $ export GEMINI_API_KEY="your-api-key-here"
2. Run the study planner:
   $ python study_planner.py
3. Available commands:
   - studyplan <path_to_syllabus> [weeks] : Create study plan PDF & optionally add to Google/Outlook Calendar
   - exit                                : Exit the study planner

Example Usage
-------------
$ python study_planner.py
> studyplan ssyllabus.pdf 16
> exit
"""

import os
import asyncio
import json
import datetime
from pathlib import Path
from typing import Optional, List
from arcadepy import Arcade
import google.generativeai as genai
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import anthropic

class StudyPlanner:
    # Supported models registry
    MODEL_REGISTRY = {
        "gemini-2.0-pro": {
            "provider": "google",
            "display_name": "Gemini 2.0 Pro",
            "model_id": "models/gemini-2.0-pro"
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "display_name": "Gemini 2.0 Flash",
            "model_id": "models/gemini-2.0-flash"
        },
        "claude-3.5-sonnet": {
            "provider": "anthropic",
            "display_name": "Claude 3.5 Sonnet",
            "model_id": "claude-3.5-sonnet-20240620"
        },
        "claude-3.7-sonnet": {
            "provider": "anthropic",
            "display_name": "Claude 3.7 Sonnet",
            "model_id": "claude-3.7-sonnet-20240620"
        }
    }

    def __init__(self, api_key: str, claude_api_key: str = None):
        # Default to Gemini 2.0 Flash
        self.current_model = "gemini-2.0-flash"
        self.api_key = api_key
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")
        self.study_plans_dir = Path("study_plans")
        self.study_plans_dir.mkdir(exist_ok=True)
        self.arcade_client = Arcade()
        self.user_id = None
        self.stored_email_from_config = None
        self.auth_file_path = Path("auth_config.json")
        self._load_auth_info()

    def _load_auth_info(self):
        try:
            if self.auth_file_path.exists():
                with open(self.auth_file_path, 'r') as f:
                    auth_data = json.load(f)
                    if isinstance(auth_data, dict):
                        self.stored_email_from_config = auth_data
                    else:
                        self.stored_email_from_config = {}
        except Exception:
            self.stored_email_from_config = {}

    def _save_auth_info(self, email: str, calendar_type: str):
        try:
            auth_data = {}
            if self.auth_file_path.exists():
                with open(self.auth_file_path, 'r') as f:
                    try:
                        auth_data = json.load(f)
                    except Exception:
                        auth_data = {}
            auth_data[calendar_type] = email
            with open(self.auth_file_path, 'w') as f:
                json.dump(auth_data, f)
        except Exception:
            pass

    async def _ensure_calendar_auth(self, calendar_type: str, input_email: str) -> bool:
        tool_name = "Google.CreateEvent" if calendar_type == "gmail" else "OutlookCalendar.CreateEvent"
        if hasattr(self, 'user_id') and getattr(self, 'user_id', None) == input_email:
            return True
        print(f"Attempting {calendar_type.title()} Calendar authorization for {input_email}...")
        auth_response = self.arcade_client.tools.authorize(tool_name=tool_name, user_id=input_email)
        if auth_response.status != "completed":
            print(f"Click this link to authorize {input_email}: {auth_response.url}")
            auth_response = self.arcade_client.auth.wait_for_completion(auth_response)
        if auth_response.status == "completed":
            print(f"Successfully authenticated {input_email} with {calendar_type.title()} Calendar.")
            self.user_id = input_email
            self._save_auth_info(input_email, calendar_type)
            if not self.stored_email_from_config:
                self.stored_email_from_config = {}
            self.stored_email_from_config[calendar_type] = input_email
            return True
        else:
            print(f"{calendar_type.title()} Authentication failed for {input_email}: {auth_response.status}")
            self.user_id = None
            return False

    async def _extract_text_from_syllabus(self, file_path: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: Syllabus file {file_path} does not exist.")
            return None
        try:
            if path.suffix.lower() == ".pdf":
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            elif path.suffix.lower() == ".txt":
                return path.read_text()
            else:
                print(f"Unsupported syllabus file type: {path.suffix}. Please use .pdf or .txt.")
                return None
        except Exception as e:
            print(f"Error extracting text from syllabus: {str(e)}")
            return None

    def set_model(self, model_name: str) -> bool:
        if model_name in self.MODEL_REGISTRY:
            self.current_model = model_name
            if self.MODEL_REGISTRY[model_name]["provider"] == "google":
                self.model = genai.GenerativeModel(self.MODEL_REGISTRY[model_name]["model_id"])
            return True
        return False

    def get_available_models(self):
        return [
            {"name": k, "display_name": v["display_name"]}
            for k, v in self.MODEL_REGISTRY.items()
        ]

    def get_current_model(self):
        m = self.MODEL_REGISTRY[self.current_model]
        return {"name": self.current_model, "display_name": m["display_name"]}

    async def _generate_study_plan_from_syllabus(self, syllabus_text: str, semester_weeks: int = 16, existing_events: List[dict] = None) -> Optional[List[dict]]:
        model_info = self.MODEL_REGISTRY[self.current_model]
        provider = model_info["provider"]
        model_id = model_info["model_id"]
        today = datetime.date.today()
        days_until_monday = (0 - today.weekday() + 7) % 7
        if days_until_monday == 0:
            upcoming_monday = today + datetime.timedelta(days=7)
        else:
            upcoming_monday = today + datetime.timedelta(days=days_until_monday)
        start_date_str = upcoming_monday.strftime("%Y-%m-%d")
        
        # Build existing events context
        existing_events_context = ""
        if existing_events:
            existing_events_context = f"""
        
        IMPORTANT: The user already has the following events in their calendar. You MUST schedule all new study events to avoid conflicts with these existing events. Do not schedule any study events at the same time as existing events:

        Existing Calendar Events:
        """
            for event in existing_events:
                try:
                    start_dt = datetime.datetime.fromisoformat(event["start_datetime"].replace("Z", "+00:00"))
                    end_dt = datetime.datetime.fromisoformat(event["end_datetime"].replace("Z", "+00:00"))
                    existing_events_context += f"""
        - "{event['summary']}" on {start_dt.strftime('%A, %B %d, %Y')} from {start_dt.strftime('%I:%M %p')} to {end_dt.strftime('%I:%M %p')}"""
                except (ValueError, KeyError):
                    continue
            
            existing_events_context += f"""

        SCHEDULING RULES:
        1. NEVER schedule study events that overlap with existing events
        2. Leave at least 30 minutes buffer before and after existing events
        3. If a preferred time slot conflicts with existing events, choose an alternative time
        4. Prefer morning hours (9 AM - 12 PM) or afternoon hours (1 PM - 5 PM) for study sessions
        5. Avoid scheduling during typical meal times (12-1 PM, 6-7 PM) unless necessary
        6. Weekend study sessions are acceptable if weekdays are too busy
        """

        prompt = f"""
        Based on the following syllabus text, create a detailed study plan for a semester of {semester_weeks} weeks.
        The study plan should break down topics, assignments, and exams week by week.
        Return the plan as a JSON array, where each item is an event with 'summary' (max 60 chars), 'description' (detailed), 
        'start_datetime', and 'end_datetime' (use ISO 8601 format like YYYY-MM-DDTHH:MM:SS).
        Assume the semester starts next Monday. For each week, create a general \"Study [Topic]\" event for 2 hours.
        If there are specific assignments or exams, create separate events for them.

        Syllabus Text:
        {syllabus_text}
        {existing_events_context}

        JSON Output Example:
        [
            {{
                "summary": "Study Ch. 1: Intro to Topic",
                "description": "Review lecture notes and read Chapter 1. Focus on key concepts A, B, C.",
                "start_datetime": "YYYY-MM-DDTHH:MM:SS", 
                "end_datetime": "YYYY-MM-DDTHH:MM:SS"
            }},
            {{
                "summary": "Assignment 1 Due",
                "description": "Submit Assignment 1 covering material from Chapters 1-2.",
                "start_datetime": "YYYY-MM-DDTHH:MM:SS",
                "end_datetime": "YYYY-MM-DDTHH:MM:SS" 
            }}
        ]

        Begin the plan from the upcoming Monday. The upcoming Monday starts on {start_date_str} (YYYY-MM-DD). For weekly study blocks, schedule them for a consistent day/time, e.g., Monday 10:00 AM. 
        For assignments/exams, use reasonable due dates/times if not specified, e.g., end of the day (17:00).
        Calculate dates based on the upcoming Monday and the week number. Ensure all datetimes are absolute and in ISO 8601 format.
        Make sure the 'summary' is concise for calendar readability. The 'description' can be more verbose.
        
        CRITICAL: If existing events were provided above, ensure NO study events conflict with them. Smart scheduling is essential.
        """
        try:
            if provider == "google":
                model = genai.GenerativeModel(model_id)
                response = await asyncio.to_thread(model.generate_content, prompt)
                cleaned_response_text = response.text.strip()
            elif provider == "anthropic":
                if not self.claude_api_key:
                    raise ValueError("Claude API key not set. Please set the CLAUDE_API_KEY environment variable.")
                client = anthropic.Anthropic(api_key=self.claude_api_key)
                response = await asyncio.to_thread(
                    lambda: client.messages.create(
                        model=model_id,
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                # Join all text blocks
                cleaned_response_text = " ".join([block.text for block in response.content if hasattr(block, "text")]).strip()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            study_plan_events = json.loads(cleaned_response_text)
            return study_plan_events
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {str(e)}")
            print(f"LLM Raw Response: {cleaned_response_text}")
            return None
        except Exception as e:
            print(f"Error generating study plan: {str(e)}")
            return None

    def _generate_study_plan_pdf(self, study_plan_events: List[dict], syllabus_file_name: str) -> Optional[str]:
        pdf_file_name = f"{Path(syllabus_file_name).stem}_study_plan.pdf"
        pdf_path = self.study_plans_dir / pdf_file_name
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        title = Paragraph(f"Study Plan for {Path(syllabus_file_name).stem}", styles['h1'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        data = [["Date", "Time", "Summary", "Description"]]
        try:
            sorted_events = sorted(study_plan_events, key=lambda x: datetime.datetime.fromisoformat(x["start_datetime"].replace("Z", "+00:00")))
        except (KeyError, ValueError):
            sorted_events = study_plan_events
        for event in sorted_events:
            try:
                start_dt_str = event.get("start_datetime", "N/A")
                end_dt_str = event.get("end_datetime", "N/A")
                summary = event.get("summary", "N/A")
                description = event.get("description", "N/A")
                start_dt = datetime.datetime.fromisoformat(start_dt_str.replace("Z", "+00:00"))
                date_str = start_dt.strftime("%Y-%m-%d (%a)")
                time_str = f"{start_dt.strftime('%I:%M %p')} - {datetime.datetime.fromisoformat(end_dt_str.replace('Z', '+00:00')).strftime('%I:%M %p')}"
                data.append([
                    Paragraph(date_str, styles['Normal']), 
                    Paragraph(time_str, styles['Normal']), 
                    Paragraph(summary, styles['Normal']), 
                    Paragraph(description, styles['Normal'])
                ])
            except (ValueError, KeyError) as e:
                data.append([Paragraph("Error", styles['Normal']), Paragraph("Invalid Event Data", styles['Normal']), Paragraph(str(event.get('summary', 'N/A')), styles['Normal']), Paragraph(str(e), styles['Normal'])])
        table = Table(data, colWidths=[1.5*inch, 2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(table)
        try:
            doc.build(story)
            return str(pdf_path)
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return None

    async def _add_event_to_calendar(self, event_data: dict, calendar_type: str) -> bool:
        if not self.user_id:
            print("User not authenticated. Cannot add event to calendar.")
            return False
        tool_name = "Google.CreateEvent" if calendar_type == "gmail" else "OutlookCalendar.CreateEvent"
        try:
            auth_response = self.arcade_client.tools.authorize(tool_name=tool_name, user_id=self.user_id)
            if auth_response.status != "completed":
                print(f"Click this link to authorize: {auth_response.url}")
                auth_response = self.arcade_client.auth.wait_for_completion(auth_response)
            if auth_response.status != "completed":
                print(f"{calendar_type.title()} Authentication failed: {auth_response.status}")
                return False
            tool_response = self.arcade_client.tools.execute(
                tool_name=tool_name,
                input=event_data,
                user_id=self.user_id,
            )
            if tool_response.status == "success" or getattr(tool_response, 'id', None):
                print(f"Successfully created event: {event_data.get('summary', event_data.get('subject', ''))}")
                return True
            else:
                print(f"Failed to create event '{event_data.get('summary', event_data.get('subject', ''))}'. Response: {tool_response}")
                return False
        except Exception as e:
            print(f"Error adding event to calendar: {str(e)}")
            return False

    async def create_study_plan(self, syllabus_file_path: str, semester_weeks: int = 16, return_data: bool = False):
        print(f"Extracting text from syllabus: {syllabus_file_path}...")
        syllabus_text = await self._extract_text_from_syllabus(syllabus_file_path)
        if not syllabus_text:
            if return_data:
                return {"success": False, "error": "Could not extract text from syllabus"}
            return False
        
        print("Generating study plan with AI. This may take a moment...")
        study_plan_events = await self._generate_study_plan_from_syllabus(syllabus_text, semester_weeks)
        if not study_plan_events:
            if return_data:
                return {"success": False, "error": "Could not generate study plan events from AI"}
            print("Could not generate study plan events from AI.")
            return False
        
        print(f"Generated {len(study_plan_events)} events for the study plan.")
        
        print("Generating PDF of the study plan...")
        pdf_file_generated_path = self._generate_study_plan_pdf(study_plan_events, Path(syllabus_file_path).name)
        if pdf_file_generated_path:
            print(f"Study plan PDF generated successfully: {pdf_file_generated_path}")
        else:
            print("Failed to generate study plan PDF.")
            if not study_plan_events:
                if return_data:
                    return {"success": False, "error": "Failed to generate PDF and no events available"}
                return False
        
        # If return_data is True, return structured data for API/frontend use
        if return_data:
            return {
                "success": True,
                "events": study_plan_events,
                "event_count": len(study_plan_events),
                "filename": Path(syllabus_file_path).name,
                "pdf_path": pdf_file_generated_path,
                "pdf_available": bool(pdf_file_generated_path)
            }
        
        # Original terminal-based flow continues here...
        user_choice = input("Would you like to add these events to your calendar? (yes/no): ").strip().lower()
        if user_choice == 'yes':
            calendar_type = input("Which calendar do you want to add the events to? (gmail/outlook): ").strip().lower()
            if calendar_type not in ["gmail", "outlook"]:
                print("Invalid choice. Please enter 'gmail' or 'outlook'.")
                return bool(pdf_file_generated_path)
            input_email = input(f"Enter your {calendar_type.title()} email address to authenticate/confirm with Arcade.dev: ").strip()
            already_authed = self.stored_email_from_config and self.stored_email_from_config.get(calendar_type) == input_email
            if not already_authed:
                if not await self._ensure_calendar_auth(calendar_type, input_email):
                    print(f"{calendar_type.title()} authentication failed. Cannot add events to calendar.")
                    return False
            else:
                self.user_id = input_email
        
            # Use the new method for adding events
            result = await self.add_study_events_to_calendar(study_plan_events, calendar_type, input_email)
            print(result["message"])
            return result["success"] or bool(pdf_file_generated_path)
        else:
            print("Skipping calendar export based on user choice.")
            return bool(pdf_file_generated_path)

    async def _fetch_existing_events(self, calendar_type: str, start_date: datetime.date, end_date: datetime.date) -> List[dict]:
        """Fetch existing calendar events for the preview period"""
        if not self.user_id:
            print("User not authenticated. Cannot fetch existing events.")
            return []
        
        # Use the correct tool name from Arcade documentation
        tool_name = "GoogleCalendar.ListEvents" if calendar_type == "gmail" else "OutlookCalendar.ListEventsInTimeRange"
        print(f"DEBUG: Attempting to fetch events using tool: {tool_name}")
        
        try:
            # Ensure authentication
            print(f"DEBUG: Authorizing tool {tool_name} for user {self.user_id}")
            auth_response = self.arcade_client.tools.authorize(tool_name=tool_name, user_id=self.user_id)
            print(f"DEBUG: Auth response status: {auth_response.status}")
            
            if auth_response.status != "completed":
                print(f"Click this link to authorize: {auth_response.url}")
                auth_response = self.arcade_client.auth.wait_for_completion(auth_response)
            
            if auth_response.status != "completed":
                print(f"{calendar_type.title()} Authentication failed: {auth_response.status}")
                return []
            
            print(f"DEBUG: Authentication successful for {tool_name}")
            
            # Prepare parameters for listing events using correct parameter names
            if calendar_type == "gmail":
                # Use correct parameter names from Arcade documentation
                list_params = {
                    "calendar_id": "primary",  # Optional
                    "min_end_datetime": start_date.isoformat() + "T00:00:00",  # Required
                    "max_start_datetime": end_date.isoformat() + "T23:59:59",  # Required
                    "max_results": 50  # Optional
                }
            else:  # Outlook - using OutlookCalendar.ListEventsInTimeRange parameters
                list_params = {
                    "start_date_time": start_date.isoformat() + "T00:00:00",  # Required - ISO 8601 format
                    "end_date_time": end_date.isoformat() + "T23:59:59",    # Required - ISO 8601 format
                    "limit": 100  # Optional - max number of events to return (defaults to 10)
                }
            
            print(f"DEBUG: List params: {list_params}")
            print(f"DEBUG: Date range: {start_date} to {end_date}")
            
            # Execute the list events tool
            tool_response = self.arcade_client.tools.execute(
                tool_name=tool_name,
                input=list_params,
                user_id=self.user_id,
            )
            
            print(f"DEBUG: Tool response status: {tool_response.status}")
            print(f"DEBUG: Tool response type: {type(tool_response)}")
            print(f"DEBUG: Tool response attributes: {dir(tool_response)}")
            
            if tool_response.status == "success":
                events = []
                # Parse response based on the tool's output format
                response_data = getattr(tool_response, 'output', tool_response)
                print(f"DEBUG: Response data type: {type(response_data)}")
                print(f"DEBUG: Response data: {response_data}")
                
                if calendar_type == "gmail":
                    # Handle Arcade's Output object format
                    if hasattr(response_data, 'value') and isinstance(response_data.value, dict):
                        actual_data = response_data.value
                        print(f"DEBUG: Found value in Output object: {actual_data}")
                        
                        # Check for events in the 'events' key (Arcade format)
                        if 'events' in actual_data:
                            event_list = actual_data['events']
                            print(f"DEBUG: Found {len(event_list)} events in response")
                            
                            for i, event in enumerate(event_list):
                                print(f"DEBUG: Event {i}: {event}")
                                events.append({
                                    "summary": event.get("summary", "No Title"),
                                    "start_datetime": event.get("start", {}).get("dateTime", event.get("start", {}).get("date", "")),
                                    "end_datetime": event.get("end", {}).get("dateTime", event.get("end", {}).get("date", "")),
                                    "description": event.get("description", ""),
                                    "is_existing": True
                                })
                        else:
                            print("DEBUG: No 'events' key found in response value")
                    
                    # Fallback: Google Calendar API format with 'items'
                    elif isinstance(response_data, dict):
                        print(f"DEBUG: Response data keys: {list(response_data.keys())}")
                        if 'items' in response_data:
                            print(f"DEBUG: Found {len(response_data['items'])} items in response")
                            for i, event in enumerate(response_data['items']):
                                print(f"DEBUG: Event {i}: {event}")
                                events.append({
                                    "summary": event.get("summary", "No Title"),
                                    "start_datetime": event.get("start", {}).get("dateTime", event.get("start", {}).get("date", "")),
                                    "end_datetime": event.get("end", {}).get("dateTime", event.get("end", {}).get("date", "")),
                                    "description": event.get("description", ""),
                                    "is_existing": True
                                })
                        else:
                            print("DEBUG: No 'items' key found in response")
                    else:
                        print(f"DEBUG: Unexpected response format: {response_data}")
                else:
                    # Outlook Calendar response format - handle both nested Output object and direct dict
                    if hasattr(response_data, 'value') and isinstance(response_data.value, dict):
                        actual_data = response_data.value
                        print(f"DEBUG: Found value in Outlook Output object: {actual_data}")
                        
                        # Check for events in the response
                        if 'events' in actual_data:
                            event_list = actual_data['events']
                            print(f"DEBUG: Found {len(event_list)} Outlook events in response")
                            
                            for i, event in enumerate(event_list):
                                print(f"DEBUG: Outlook Event {i}: {event}")
                                events.append({
                                    "summary": event.get("subject", "No Title"),
                                    "start_datetime": event.get("start", {}).get("dateTime", "") if isinstance(event.get("start"), dict) else event.get("start", ""),
                                    "end_datetime": event.get("end", {}).get("dateTime", "") if isinstance(event.get("end"), dict) else event.get("end", ""),
                                    "description": event.get("body", {}).get("content", "") if isinstance(event.get("body"), dict) else str(event.get("body", "")),
                                    "is_existing": True
                                })
                        elif 'value' in actual_data:
                            # Alternative format with 'value' array
                            for event in actual_data['value']:
                                events.append({
                                    "summary": event.get("subject", "No Title"),
                                    "start_datetime": event.get("start", {}).get("dateTime", "") if isinstance(event.get("start"), dict) else event.get("start", ""),
                                    "end_datetime": event.get("end", {}).get("dateTime", "") if isinstance(event.get("end"), dict) else event.get("end", ""),
                                    "description": event.get("body", {}).get("content", "") if isinstance(event.get("body"), dict) else str(event.get("body", "")),
                                    "is_existing": True
                                })
                        else:
                            print("DEBUG: No 'events' or 'value' key found in Outlook response")
                    
                    # Fallback: direct dict format
                    elif isinstance(response_data, dict):
                        print(f"DEBUG: Outlook response data keys: {list(response_data.keys())}")
                        if 'value' in response_data:
                            print(f"DEBUG: Found {len(response_data['value'])} Outlook items in response")
                            for i, event in enumerate(response_data['value']):
                                print(f"DEBUG: Outlook Event {i}: {event}")
                                events.append({
                                    "summary": event.get("subject", "No Title"),
                                    "start_datetime": event.get("start", {}).get("dateTime", "") if isinstance(event.get("start"), dict) else event.get("start", ""),
                                    "end_datetime": event.get("end", {}).get("dateTime", "") if isinstance(event.get("end"), dict) else event.get("end", ""),
                                    "description": event.get("body", {}).get("content", "") if isinstance(event.get("body"), dict) else str(event.get("body", "")),
                                    "is_existing": True
                                })
                        else:
                            print("DEBUG: No 'value' key found in Outlook response")
                    else:
                        print(f"DEBUG: Unexpected Outlook response format: {response_data}")
                
                print(f"DEBUG: Processed {len(events)} existing events from {calendar_type} calendar")
                return events
            else:
                print(f"DEBUG: Tool execution failed. Status: {tool_response.status}")
                print(f"DEBUG: Full tool response: {tool_response}")
                return []
                
        except Exception as e:
            print(f"DEBUG: Exception with tool {tool_name}: {str(e)}")
            print(f"DEBUG: Full error details: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def get_calendar_preview_data(self, study_plan_events: List[dict], calendar_type: str, input_email: str) -> dict:
        """Get combined calendar preview data with existing and projected events"""
        try:
            # Set user_id for authentication
            self.user_id = input_email
            
            # Ensure authentication
            if not await self._ensure_calendar_auth(calendar_type, input_email):
                return {
                    "success": False,
                    "error": f"{calendar_type.title()} authentication failed",
                    "existing_events": [],
                    "study_events": study_plan_events
                }
            
            # Calculate date range for fetching existing events
            start_dates = []
            end_dates = []
            for event in study_plan_events:
                try:
                    start_dt = datetime.datetime.fromisoformat(event["start_datetime"].replace("Z", "+00:00"))
                    end_dt = datetime.datetime.fromisoformat(event["end_datetime"].replace("Z", "+00:00"))
                    start_dates.append(start_dt.date())
                    end_dates.append(end_dt.date())
                except (KeyError, ValueError):
                    continue
            
            if start_dates and end_dates:
                preview_start = min(start_dates) - datetime.timedelta(days=7)  # Include week before
                preview_end = max(end_dates) + datetime.timedelta(days=7)    # Include week after
            else:
                # Fallback to reasonable range
                today = datetime.date.today()
                preview_start = today
                preview_end = today + datetime.timedelta(weeks=20)
            
            # Fetch existing events
            existing_events = await self._fetch_existing_events(calendar_type, preview_start, preview_end)
            
            return {
                "success": True,
                "existing_events": existing_events,
                "study_events": study_plan_events,
                "date_range": {
                    "start": preview_start.isoformat(),
                    "end": preview_end.isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "existing_events": [],
                "study_events": study_plan_events
            }

    def format_calendar_events_for_display(self, existing_events: List[dict], study_events: List[dict]) -> List[dict]:
        """Format and combine events for frontend display"""
        all_events = []
        
        # Add existing events with formatting
        for event in existing_events:
            try:
                start_dt = datetime.datetime.fromisoformat(event["start_datetime"].replace("Z", "+00:00"))
                end_dt = datetime.datetime.fromisoformat(event["end_datetime"].replace("Z", "+00:00"))
                
                all_events.append({
                    "id": f"existing_{hash(event['summary'] + event['start_datetime'])}",
                    "title": event.get("summary", "No Title"),
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "description": event.get("description", ""),
                    "type": "existing",
                    "color": "#4285f4",  # Google blue
                    "textColor": "white"
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed existing event: {e}")
                continue
        
        # Add projected study events with formatting
        for event in study_events:
            try:
                start_dt = datetime.datetime.fromisoformat(event["start_datetime"].replace("Z", "+00:00"))
                end_dt = datetime.datetime.fromisoformat(event["end_datetime"].replace("Z", "+00:00"))
                
                all_events.append({
                    "id": f"projected_{hash(event['summary'] + event['start_datetime'])}",
                    "title": event.get("summary", "No Title"),
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "description": event.get("description", ""),
                    "type": "projected",
                    "color": "#34a853",  # Green
                    "textColor": "white"
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed study event: {e}")
                continue
        
        # Sort events by start time
        all_events.sort(key=lambda x: x["start"])
        
        return all_events

    async def add_study_events_to_calendar(self, study_plan_events: List[dict], calendar_type: str, input_email: str) -> dict:
        """Add study plan events to user's calendar and return results"""
        try:
            # Set user_id
            self.user_id = input_email
            
            # Process events for calendar format
            processed_calendar_events = []
            for event_template in study_plan_events:
                try:
                    if calendar_type == "gmail":
                        # Validate datetime format
                        datetime.datetime.fromisoformat(event_template["start_datetime"].replace("Z", "+00:00"))
                        datetime.datetime.fromisoformat(event_template["end_datetime"].replace("Z", "+00:00"))
                        
                        calendar_event_input = {
                            "calendar_id": "primary",
                            "summary": event_template["summary"],
                            "description": event_template.get("description", ""),
                            "start_datetime": event_template["start_datetime"],
                            "end_datetime": event_template["end_datetime"],
                        }
                    else:  # Outlook
                        calendar_event_input = {
                            "subject": event_template["summary"],
                            "body": event_template.get("description", ""),
                            "start_date_time": event_template["start_datetime"].split(".")[0],
                            "end_date_time": event_template["end_datetime"].split(".")[0],
                        }
                    processed_calendar_events.append(calendar_event_input)
                except KeyError as e:
                    print(f"Skipping calendar event due to missing key: {e}. Event data: {event_template}")
                except ValueError as e:
                    print(f"Skipping calendar event due to invalid datetime format: {e}. Event data: {event_template}")
            
            if not processed_calendar_events:
                return {
                    "success": False,
                    "error": "No valid events to add to calendar after processing",
                    "added_events": 0,
                    "total_events": len(study_plan_events)
                }
            
            # Add events to calendar
            successful_adds = 0
            failed_events = []
            
            for event_data in processed_calendar_events:
                if await self._add_event_to_calendar(event_data, calendar_type):
                    successful_adds += 1
                else:
                    failed_events.append(event_data.get('summary', event_data.get('subject', 'Unknown Event')))
            
            return {
                "success": True,
                "added_events": successful_adds,
                "total_events": len(processed_calendar_events),
                "failed_events": failed_events,
                "message": f"Successfully added {successful_adds} out of {len(processed_calendar_events)} events to your {calendar_type.title()} Calendar."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "added_events": 0,
                "total_events": len(study_plan_events)
            }

    async def create_smart_study_plan(self, syllabus_file_path: str, semester_weeks: int = 16, calendar_type: str = None, input_email: str = None):
        """Create a study plan that intelligently avoids existing calendar events"""
        print(f"Creating smart study plan from: {syllabus_file_path}...")
        
        # Extract syllabus text
        syllabus_text = await self._extract_text_from_syllabus(syllabus_file_path)
        if not syllabus_text:
            return {"success": False, "error": "Could not extract text from syllabus"}
        
        existing_events = []
        
        # Fetch existing events if calendar info provided
        if calendar_type and input_email:
            print(f"Fetching existing events from {calendar_type} calendar...")
            self.user_id = input_email
            
            # Calculate date range for existing events (semester duration + buffer)
            today = datetime.date.today()
            start_date = today - datetime.timedelta(days=7)
            end_date = today + datetime.timedelta(weeks=semester_weeks + 4)
            
            try:
                existing_events = await self._fetch_existing_events(calendar_type, start_date, end_date)
                print(f"Found {len(existing_events)} existing events to work around")
            except Exception as e:
                print(f"Warning: Could not fetch existing events: {e}")
                # Continue without existing events context
        
        # Generate study plan with existing events context
        print("Generating AI study plan with smart scheduling...")
        study_plan_events = await self._generate_study_plan_from_syllabus(
            syllabus_text, 
            semester_weeks, 
            existing_events
        )
        
        if not study_plan_events:
            return {"success": False, "error": "Could not generate study plan events from AI"}
        
        print(f"Generated {len(study_plan_events)} study events that avoid conflicts")
        
        # Generate PDF
        pdf_file_generated_path = self._generate_study_plan_pdf(study_plan_events, Path(syllabus_file_path).name)
        
        return {
            "success": True,
            "events": study_plan_events,
            "event_count": len(study_plan_events),
            "filename": Path(syllabus_file_path).name,
            "pdf_path": pdf_file_generated_path,
            "pdf_available": bool(pdf_file_generated_path),
            "existing_events_considered": len(existing_events),
            "smart_scheduling": bool(existing_events)
        }

    def get_study_plan_with_calendar_preview(self, study_plan_events: List[dict], syllabus_file_name: str) -> dict:
        """Return study plan data formatted for frontend with calendar preview support"""
        return {
            "success": True,
            "events": study_plan_events,
            "event_count": len(study_plan_events),
            "filename": syllabus_file_name,
            "pdf_available": True,  # Assuming PDF was generated
            "calendar_preview_available": True
        }

    async def create_study_plan_for_api(self, syllabus_file_path: str, semester_weeks: int = 16):
        """
        Create study plan for API use - returns (pdf_path, events) tuple instead of CLI prompts
        """
        print(f"Extracting text from syllabus: {syllabus_file_path}...")
        syllabus_text = await self._extract_text_from_syllabus(syllabus_file_path)
        if not syllabus_text:
            print("Failed to extract text from syllabus")
            return None, []

        print("Generating study plan with AI. This may take a moment...")
        study_plan_events = await self._generate_study_plan_from_syllabus(syllabus_text, semester_weeks)
        if not study_plan_events:
            print("Could not generate study plan events from AI.")
            return None, []

        print(f"Generated {len(study_plan_events)} events for the study plan.")

        print("Generating PDF of the study plan...")
        pdf_file_generated_path = self._generate_study_plan_pdf(study_plan_events, Path(syllabus_file_path).name)
        if not pdf_file_generated_path:
            print("Failed to generate study plan PDF.")
            return None, []

        print(f"Study plan PDF generated successfully: {pdf_file_generated_path}")
        print(f"Returning PDF path: {pdf_file_generated_path}")
        print(f"Returning events count: {len(study_plan_events)}")
        
        return (pdf_file_generated_path, study_plan_events)

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    planner = StudyPlanner(api_key)
    print("\nWelcome to Med-Bot Study Planner!")
    print("----------------------------------")
    print("Commands:")
    print("  studyplan <path_to_syllabus> [number_of_semester_weeks] - Create study plan PDF & optionally add to Google/Outlook Calendar")
    print("  exit - Exit the study planner")
    print("  models - List available models")
    print("  model <model_name> - Switch to a different model")
    print("----------------------------------\n")
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower().startswith("studyplan "):
                parts = user_input.split(" ")
                if len(parts) >= 2:
                    syllabus_file_path = parts[1].strip()
                    semester_weeks = 16
                    if len(parts) > 2 and parts[2].isdigit():
                        semester_weeks = int(parts[2])
                    print(f"\nAttempting to create study plan from: {syllabus_file_path} for {semester_weeks} weeks...")
                    await planner.create_study_plan(syllabus_file_path, semester_weeks)
                else:
                    print("Usage: studyplan <path_to_syllabus_file> [number_of_semester_weeks]")
            elif user_input.lower() == "models":
                print("\nAvailable models:")
                for m in planner.get_available_models():
                    print(f"  {m['name']}: {m['display_name']}")
                current = planner.get_current_model()
                print(f"\nCurrent model: {current['name']} ({current['display_name']})")
            elif user_input.lower().startswith("model "):
                model_name = user_input[6:].strip()
                if planner.set_model(model_name):
                    current = planner.get_current_model()
                    print(f"Switched to model: {current['name']} ({current['display_name']})")
                else:
                    print(f"Model '{model_name}' not found. Use 'models' to list available models.")
            else:
                print("Unknown command. Available commands: studyplan, exit, models, model")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 