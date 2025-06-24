from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.ext.asyncio import AsyncSession
from .state import RedditAgentInput, AgentState
from .graph import reddit_graph
from app.database import get_db
import structlog

router = APIRouter()
logger = structlog.get_logger()

@router.post("/reddit/reddit-agent", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def run_reddit_agent(input: RedditAgentInput, db: AsyncSession = Depends(get_db)):
    try:
        initial_state = AgentState(
            agent_name=input.agent_name,
            goals=input.goals,
            instructions=input.instructions,
            description=input.description,
            expectation=input.expectation,
            target_audience=input.target_audience,
            company_keywords=input.company_keywords,
            min_upvotes=input.min_upvotes,
            max_age_days=input.max_age_days,
            restrict_to_goal_subreddits=input.restrict_to_goal_subreddits,
            subreddits=["intune", "mdm"],
            # subreddits=["intune", "mdm", "sysadmin", "airwatchmdm", "windowssecurity", "miradore", "theinternetofshit", "iot", "technology", "iotsecurity", "iotmostuseful", "filewaveuem", "blackberryuem", "uem", "saasblogs", "endpoint", "threatintel", "action1", "secnewsgr", "enterprisesecurity", "securityautomation", "securednews", "enterpriseit", "storage", "networking", "iosinfosec", "ios", "applesecurity", "zygosec", "databreaches", "databreach", "pwned", "indiatech", "databreached", "datasecurity", "iso13485", "oktasso", "coding", "suremdm", "hexnode_mdm", "macsysadmin", "mdm_solution", "msp360", "thingsboard", "hexnodeuem", "altiris", "citrix", "automox", "netsec", "mobileiron", "asknetsec", "ninjarmm", "optitune", "cybersecurity", "sysadminjobs", "securityctf", "securityit", "it_securitylabs", "computersecurity", "security", "talesfromsecurity", "connectusasmbs", "threatinformeddefense", "opsec", "cybersecurity_", "mobilerepair", "mobiledeviceforensics", "msp", "k12sysadmin", "cissp", "smallmsp", "mspjobs", "socialmedia", "digital_marketing", "businessanalysis", "facebookads", "softwarearchitecture", "digitalsignage", "digitalsigns", "xibo", "pads4", "commercialav", "sccm", "azure", "msintune", "it_management", "itcareerquestions", "projectmanagement", "productmanagement", "management", "appleenterprise", "android", "androiddev", "vmware", "workspaceone", "biztalk", "freshservice", "ipad", "appsense", "ecommerce_growth", "mobilesecurity", "oem", "compliance", "remotework", "remoteworking", "cisoseries", "cybersecuk", "ravendpointprotection", "walletscrutiny", "structuralengineering", "bitwarden", "filewave", "android_security", "androidquestions", "andsec", "droidsec", "defenderatp", "pming", "framrtv", "posterbooking", "tightropemediasystems", "mobsec", "grapheneos", "remoteteammanagement", "saltstack", "rmmcomparison", "rmmreviews", "vmwareworkspaceone", "omnissaeuc", "cloudarchitect", "cloudfirstmsp", "waas", "managed_services", "telin", "techsupport", "windows10", "win"],
            keywords=[],
            posts=[],
            retries=0,
            error=None,
            db=db,
            llm=None
        )
        result = await reddit_graph.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error("Reddit agent processing failed", agent_name=input.agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 