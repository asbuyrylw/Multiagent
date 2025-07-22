from agents.base_agent import BaseAgent
from tools.redfin_data_tool import get_redfin_data, get_property_details
from tools.census_data_tool import get_census_data, combine_census_data
from utils.logger import setup_logger

class HousingDataAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.logger = setup_logger(f"housing_data_agent_{name}")

    def handle(self, task):
        """
        Handle housing data collection tasks.
        """
        self.logger.info(f"HousingDataAgent handling task: {task}")
        
        location = task.get('location', 'Seattle, WA')
        property_count = task.get('property_count', 100)
        
        try:
            # Get Redfin property data
            self.logger.info(f"Collecting Redfin data for {location}")
            property_data = get_redfin_data(location, property_count)
            
            # Get Census demographic data
            self.logger.info(f"Collecting Census data for {location}")
            census_data = combine_census_data(location)
            
            # Get additional property details for a sample
            sample_properties = property_data.head(10)['property_id'].tolist()
            detailed_data = []
            for prop_id in sample_properties:
                details = get_property_details(prop_id)
                detailed_data.append(details)
            
            result = {
                'status': 'success',
                'location': location,
                'property_data': property_data,
                'census_data': census_data,
                'detailed_property_sample': detailed_data,
                'summary': {
                    'total_properties': len(property_data),
                    'properties_likely_to_sell': property_data['will_sell_within_year'].sum(),
                    'avg_property_value': property_data['current_value'].mean(),
                    'area_population': census_data['total_population'],
                    'area_economic_health': census_data['economic_health']
                }
            }
            
            self.logger.info(f"Successfully collected housing data for {location}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error collecting housing data: {str(e)}")
            return {
                'status': 'error',
                'message': f"Failed to collect housing data: {str(e)}"
            }
