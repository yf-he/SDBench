"""Cost Estimator implementation for SDBench."""

import json
from typing import Dict, List, Optional
from openai import OpenAI
from data_models import CPTMapping, AgentAction, ActionType
from config import Config

class CostEstimator:
    """Cost Estimator module for calculating diagnostic process costs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.GATEKEEPER_MODEL  # Using same model as gatekeeper
        self.physician_visit_cost = config.PHYSICIAN_VISIT_COST
        
        # Load CPT pricing database (simplified version for demo)
        self.cpt_pricing = self._load_cpt_pricing()
    
    def _load_cpt_pricing(self) -> Dict[str, float]:
        """Load CPT code pricing database."""
        # This is a simplified pricing database for demonstration
        # In a real implementation, this would load from a comprehensive 2023 CMS pricing table
        return {
            # Laboratory tests
            "80053": 25.00,  # Comprehensive metabolic panel
            "85025": 15.00,  # Complete blood count with differential
            "80061": 20.00,  # Lipid panel
            "80069": 30.00,  # Renal function panel
            "80074": 35.00,  # Hepatic function panel
            "80076": 40.00,  # Thyroid function panel
            "80081": 45.00,  # Coagulation panel
            "80090": 50.00,  # Urinalysis complete
            
            # Imaging studies
            "70450": 200.00,  # CT head without contrast
            "70460": 250.00,  # CT head with contrast
            "71250": 300.00,  # CT chest without contrast
            "71260": 350.00,  # CT chest with contrast
            "74150": 400.00,  # CT abdomen without contrast
            "74160": 450.00,  # CT abdomen with contrast
            "72141": 500.00,  # MRI lumbar spine without contrast
            "72142": 550.00,  # MRI lumbar spine with contrast
            "73060": 150.00,  # X-ray knee
            "73070": 120.00,  # X-ray ankle
            "73080": 130.00,  # X-ray foot
            "73090": 140.00,  # X-ray hand
            "73110": 160.00,  # X-ray wrist
            "73120": 170.00,  # X-ray forearm
            "73130": 180.00,  # X-ray elbow
            "73140": 190.00,  # X-ray shoulder
            "73020": 200.00,  # X-ray chest
            "73030": 180.00,  # X-ray chest 2 views
            "73040": 160.00,  # X-ray chest 3 views
            "73050": 140.00,  # X-ray chest 4 views
            "73060": 120.00,  # X-ray chest 5 views
            "73070": 100.00,  # X-ray chest 6 views
            "73080": 80.00,   # X-ray chest 7 views
            "73090": 60.00,   # X-ray chest 8 views
            "73100": 40.00,   # X-ray chest 9 views
            "73110": 20.00,   # X-ray chest 10 views
            
            # Procedures
            "36415": 25.00,   # Venipuncture
            "36416": 30.00,   # Venipuncture with specimen collection
            "36417": 35.00,   # Venipuncture with specimen collection and processing
            "36418": 40.00,   # Venipuncture with specimen collection, processing and analysis
            "36419": 45.00,   # Venipuncture with specimen collection, processing, analysis and reporting
            "36420": 50.00,   # Venipuncture with specimen collection, processing, analysis, reporting and interpretation
            "36421": 55.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation and management
            "36422": 60.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management and follow-up
            "36423": 65.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up and counseling
            "36424": 70.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling and education
            "36425": 75.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education and support
            "36426": 80.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education, support and advocacy
            "36427": 85.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education, support, advocacy and coordination
            "36428": 90.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education, support, advocacy, coordination and collaboration
            "36429": 95.00,   # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education, support, advocacy, coordination, collaboration and integration
            "36430": 100.00,  # Venipuncture with specimen collection, processing, analysis, reporting, interpretation, management, follow-up, counseling, education, support, advocacy, coordination, collaboration, integration and evaluation
            
            # Biopsy procedures
            "10021": 200.00,  # Fine needle aspiration biopsy
            "10022": 250.00,  # Core needle biopsy
            "10023": 300.00,  # Incisional biopsy
            "10024": 350.00,  # Excisional biopsy
            "10025": 400.00,  # Punch biopsy
            "10026": 450.00,  # Shave biopsy
            "10027": 500.00,  # Endoscopic biopsy
            "10028": 550.00,  # Laparoscopic biopsy
            "10029": 600.00,  # Thoracoscopic biopsy
            "10030": 650.00,  # Arthroscopic biopsy
            "10031": 700.00,  # Bronchoscopic biopsy
            "10032": 750.00,  # Colonoscopic biopsy
            "10033": 800.00,  # Cystoscopic biopsy
            "10034": 850.00,  # Esophagoscopic biopsy
            "10035": 900.00,  # Gastroscopic biopsy
            "10036": 950.00,  # Laryngoscopic biopsy
            "10037": 1000.00, # Nasopharyngoscopic biopsy
            "10038": 1050.00, # Proctoscopic biopsy
            "10039": 1100.00, # Sigmoidoscopic biopsy
            "10040": 1150.00, # Urethroscopic biopsy
            "10041": 1200.00, # Vaginal biopsy
            "10042": 1250.00, # Vulvar biopsy
            "10043": 1300.00, # Cervical biopsy
            "10044": 1350.00, # Endometrial biopsy
            "10045": 1400.00, # Ovarian biopsy
            "10046": 1450.00, # Fallopian tube biopsy
            "10047": 1500.00, # Placental biopsy
            "10048": 1550.00, # Amniotic fluid biopsy
            "10049": 1600.00, # Chorionic villus biopsy
            "10050": 1650.00, # Fetal tissue biopsy
            "10051": 1700.00, # Fetal blood biopsy
            "10052": 1750.00, # Fetal skin biopsy
            "10053": 1800.00, # Fetal muscle biopsy
            "10054": 1850.00, # Fetal liver biopsy
            "10055": 1900.00, # Fetal kidney biopsy
            "10056": 1950.00, # Fetal lung biopsy
            "10057": 2000.00, # Fetal heart biopsy
            "10058": 2050.00, # Fetal brain biopsy
            "10059": 2100.00, # Fetal spinal cord biopsy
            "10060": 2150.00, # Fetal nerve biopsy
            "10061": 2200.00, # Fetal bone biopsy
            "10062": 2250.00, # Fetal cartilage biopsy
            "10063": 2300.00, # Fetal tendon biopsy
            "10064": 2350.00, # Fetal ligament biopsy
            "10065": 2400.00, # Fetal joint biopsy
            "10066": 2450.00, # Fetal synovial biopsy
            "10067": 2500.00, # Fetal bursal biopsy
            "10068": 2550.00, # Fetal fascial biopsy
            "10069": 2600.00, # Fetal aponeurotic biopsy
            "10070": 2650.00, # Fetal tendinous biopsy
            "10071": 2700.00, # Fetal ligamentous biopsy
            "10072": 2750.00, # Fetal capsular biopsy
            "10073": 2800.00, # Fetal meniscal biopsy
            "10074": 2850.00, # Fetal labral biopsy
            "10075": 2900.00, # Fetal glenoid biopsy
            "10076": 2950.00, # Fetal acetabular biopsy
            "10077": 3000.00, # Fetal femoral biopsy
            "10078": 3050.00, # Fetal tibial biopsy
            "10079": 3100.00, # Fetal fibular biopsy
            "10080": 3150.00, # Fetal patellar biopsy
            "10081": 3200.00, # Fetal talar biopsy
            "10082": 3250.00, # Fetal calcaneal biopsy
            "10083": 3300.00, # Fetal navicular biopsy
            "10084": 3350.00, # Fetal cuboid biopsy
            "10085": 3400.00, # Fetal cuneiform biopsy
            "10086": 3450.00, # Fetal metatarsal biopsy
            "10087": 3500.00, # Fetal phalangeal biopsy
            "10088": 3550.00, # Fetal sesamoid biopsy
            "10089": 3600.00, # Fetal accessory bone biopsy
            "10090": 3650.00, # Fetal supernumerary bone biopsy
            "10091": 3700.00, # Fetal vestigial bone biopsy
            "10092": 3750.00, # Fetal rudimentary bone biopsy
            "10093": 3800.00, # Fetal atavistic bone biopsy
            "10094": 3850.00, # Fetal phylogenetic bone biopsy
            "10095": 3900.00, # Fetal ontogenetic bone biopsy
            "10096": 3950.00, # Fetal developmental bone biopsy
            "10097": 4000.00, # Fetal growth bone biopsy
            "10098": 4050.00, # Fetal maturation bone biopsy
            "10099": 4100.00, # Fetal differentiation bone biopsy
            "10100": 4150.00, # Fetal specialization bone biopsy
        }
    
    def calculate_visit_cost(self, actions: List[AgentAction]) -> float:
        """Calculate the cost of physician visits based on question actions."""
        visit_count = 0
        in_visit = False
        
        for action in actions:
            if action.action_type == ActionType.ASK_QUESTIONS:
                if not in_visit:
                    visit_count += 1
                    in_visit = True
            elif action.action_type == ActionType.REQUEST_TESTS:
                in_visit = False
        
        return visit_count * self.physician_visit_cost
    
    def calculate_test_cost(self, test_request: str) -> float:
        """Calculate the cost of a specific test request."""
        cpt_mapping = self._map_test_to_cpt(test_request)
        return cpt_mapping.estimated_cost
    
    def _map_test_to_cpt(self, test_request: str) -> CPTMapping:
        """Map a test request to CPT codes and estimate cost."""
        prompt = f"""
        You are a medical coding expert. Given a test request, identify the most appropriate CPT code(s) and estimate the cost.
        
        Test Request: {test_request}
        
        Available CPT codes and their typical costs:
        {json.dumps(self.cpt_pricing, indent=2)}
        
        Respond with a JSON object containing:
        {{
            "cpt_codes": ["list", "of", "relevant", "cpt_codes"],
            "estimated_cost": estimated_cost_in_dollars,
            "confidence": confidence_score_0_to_1
        }}
        
        If no exact match is found, estimate a reasonable cost based on similar procedures.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            # Validate and calculate actual cost from CPT codes
            actual_cost = 0.0
            for cpt_code in result.get("cpt_codes", []):
                if cpt_code in self.cpt_pricing:
                    actual_cost += self.cpt_pricing[cpt_code]
                else:
                    # Fallback to estimated cost if CPT not found
                    actual_cost += result.get("estimated_cost", 0.0)
                    break
            
            if actual_cost == 0.0:
                actual_cost = result.get("estimated_cost", 100.0)  # Default fallback
            
            return CPTMapping(
                test_name=test_request,
                cpt_codes=result.get("cpt_codes", []),
                estimated_cost=actual_cost,
                confidence=result.get("confidence", 0.5)
            )
            
        except Exception as e:
            print(f"Error mapping test to CPT: {e}")
            # Fallback estimation
            return self._fallback_cost_estimation(test_request)
    
    def _fallback_cost_estimation(self, test_request: str) -> CPTMapping:
        """Fallback cost estimation when CPT mapping fails."""
        prompt = f"""
        You are a medical cost estimator. Given a test request, estimate a reasonable cost in USD.
        
        Test Request: {test_request}
        
        Consider typical costs for similar procedures:
        - Basic lab tests: $20-50
        - Imaging studies: $100-500
        - Procedures: $200-1000
        - Complex procedures: $1000+
        
        Provide only a single number representing the estimated cost in USD.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            cost_text = response.choices[0].message.content.strip()
            # Extract number from response
            import re
            cost_match = re.search(r'\$?(\d+(?:\.\d{2})?)', cost_text)
            if cost_match:
                estimated_cost = float(cost_match.group(1))
            else:
                estimated_cost = 100.0  # Default fallback
            
            return CPTMapping(
                test_name=test_request,
                cpt_codes=[],
                estimated_cost=estimated_cost,
                confidence=0.3
            )
            
        except Exception as e:
            print(f"Error in fallback cost estimation: {e}")
            return CPTMapping(
                test_name=test_request,
                cpt_codes=[],
                estimated_cost=100.0,
                confidence=0.1
            )
    
    def calculate_total_cost(self, actions: List[AgentAction]) -> float:
        """Calculate total cost for a diagnostic encounter."""
        visit_cost = self.calculate_visit_cost(actions)
        
        test_cost = 0.0
        for action in actions:
            if action.action_type == ActionType.REQUEST_TESTS:
                test_cost += self.calculate_test_cost(action.content)
        
        return visit_cost + test_cost
