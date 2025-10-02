import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.observers.loggers.user_bot_latency_log_observer import UserBotLatencyLogObserver

from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.runner.types import RunnerArguments
from pipecat_flows import NodeConfig, FlowManager, FlowResult, FlowsFunctionSchema, FlowArgs

from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from deepgram import LiveOptions

# Load environment variables
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Flow Nodes
# -----------------------------

def initial() -> NodeConfig:
    record_name_func = FlowsFunctionSchema(
        name="record_name",
        description="Record the user's name.",
        required=["name"],
        handler=handle_record_name,
        properties={"name": {"type": "string", "description": "The user's name"}},
    )

    return NodeConfig(
        name="initial",
        role_messages=[{"role": "system", "content": "You are polite and friendly."}],
        task_messages=[{"role": "system", "content": "Hello! What's your name?"}],
        functions=[record_name_func],
    )

async def handle_record_name(args: FlowArgs, flow_manager: FlowManager):
    user_name = args.get("name", "<unknown>")
    logger.info(f"User's name: {user_name}")
    next_node = availability(user_name)
    return FlowResult(status="ok", data={"name": user_name}), next_node

def availability(user_name: str) -> NodeConfig:
    select_time_func = FlowsFunctionSchema(
        name="select_time",
        description="User selects a time for appointment.",
        required=["time"],
        handler=handle_time_selection,
        properties={"time": {"type": "string", "description": "Requested time (e.g. 10am, 2pm, 5pm)"}},
    )

    return NodeConfig(
        name="availability",
        role_messages=[{"role": "system", "content": "You are a helpful assistant."}],
        task_messages=[{"role": "system", "content": f"Hi {user_name}, here are available slots: 10am, 2pm, 5pm. Which one works for you?"}],
        functions=[select_time_func],
    )

async def handle_time_selection(args: FlowArgs, flow_manager: FlowManager):
    requested_time = args.get("time")
    logger.info(f"User selected time: {requested_time}")

    available_slots = ["10am", "2pm", "5pm"]
    if requested_time not in available_slots:
        return FlowResult(status="retry", data={"reason": "Invalid slot"}), None

    confirm_node = confirm(requested_time)
    return FlowResult(status="ok", data={"time": requested_time}), confirm_node

def confirm(requested_time: str) -> NodeConfig:
    confirm_booking_func = FlowsFunctionSchema(
        name="confirm_booking",
        description="User confirms the appointment.",
        required=["confirm"],
        handler=handle_confirm_booking,
        properties={"confirm": {"type": "boolean", "description": "True if user confirms"}},
    )

    return NodeConfig(
        name="confirm",
        role_messages=[{"role": "system", "content": "You are confirming the appointment."}],
        task_messages=[{"role": "system", "content": f"You chose {requested_time}. Do you want to confirm this booking?"}],
        functions=[confirm_booking_func],
    )

async def handle_confirm_booking(args: FlowArgs, flow_manager: FlowManager):
    confirm = args.get("confirm", False)

    if confirm:
        logger.info("Booking confirmed ✅")
        return FlowResult(status="ok", data={"confirmed": True}), create_end_node()
    else:
        logger.info("Booking rejected ❌")
        return FlowResult(status="rejected", data={"confirmed": False}), initial()

def create_end_node(user_name: str = None) -> NodeConfig:
    content = f"Thank you, {user_name}, for answering. Goodbye!" if user_name else "Thank you for answering. Goodbye!"
    logger.info(f"Creating end node. Content: {content}")
    return NodeConfig(
        name="end_node",
        role_messages=[{"role": "system", "content": "You are polite and concise."}],
        task_messages=[{"role": "system", "content": content}],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/voice.html")

# -----------------------------
# Bot runner
# -----------------------------

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # --- Initialize services with optimizations ---
    
    # Optimized STT with interim results for faster interruption detection
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            interim_results=True,        # Faster interruption detection
            smart_format=True,           # Better LLM comprehension
            punctuate=True,              # Add punctuation
            vad_events=False,            # Use pipeline VAD instead
        )
    )
    
    # Optimized TTS (already using Cartesia WebSocket - excellent choice)
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
        text_filters=[MarkdownTextFilter()],
    )
    
    # LLM already optimized with gpt-4o-mini (good choice for speed)
    llm = OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"), 
        model="gpt-4o-mini"
    )

    # --- Context setup ---
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # --- Build pipeline ---
    pipeline = Pipeline([
        transport.input(),          # User input
        stt,                        # Deepgram STT with optimizations
        context_aggregator.user(),  # Track user context
        llm,                        # OpenRouter LLM
        tts,                        # Cartesia TTS
        transport.output(),         # Output to user
        context_aggregator.assistant(),  # Track assistant context
    ])

    # Fully optimized task configuration
    task = PipelineTask(
        pipeline, 
        params=PipelineParams(
            # Audio optimization - consistent sample rates
            audio_in_sample_rate=16000,   # Optimal for STT
            audio_out_sample_rate=24000,  # High quality TTS output
            
            # Interruption handling
            allow_interruptions=True,
            interruption_strategies=[
                MinWordsInterruptionStrategy(min_words=3)  # Prevent unwanted interruptions
            ],
            
            # Performance monitoring
            enable_metrics=True,              # Monitor TTFB and processing times
            enable_usage_metrics=True,        # Track API usage
            report_only_initial_ttfb=True,    # Reduce log noise
            
            # Observers for detailed latency tracking
            observers=[
                UserBotLatencyLogObserver()   # Track speech-to-speech latency
            ]
        )
    )

    # --- Initialize flow manager ---
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
    
    # --- Transport event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await flow_manager.initialize(initial())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # --- Run the pipeline ---
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

# -------------------
# WebRTC offer endpoint
# -------------------
@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """
    WebRTC offer/answer endpoint.
    request should contain {"sdp": ..., "type": ...} from the browser
    """
    # Create WebRTC connection
    webrtc_connection = SmallWebRTCConnection()
    await webrtc_connection.initialize(sdp=request["sdp"], type=request["type"])

    # Optimized transport with local VAD for 150-200ms faster speech detection
    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),  # 150-200ms faster than remote VAD
        ),
        webrtc_connection=webrtc_connection,
    )

    # Run your bot in background
    background_tasks.add_task(run_bot, transport, RunnerArguments())

    # Return WebRTC answer
    answer = webrtc_connection.get_answer()
    return {"sdp": answer["sdp"], "type": answer["type"]}

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Pipecat SmallWebRTC server is running"}

# -----------------------------
# Run FastAPI server
# -----------------------------
if __name__ == "__main__":
    required_env_vars = ["DEEPGRAM_API_KEY", "OPENROUTER_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        exit(1)

    import uvicorn
    logger.info("Starting Pipecat SmallWebRTC FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")