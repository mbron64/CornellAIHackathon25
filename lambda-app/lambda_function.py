from thisapp.aws_lambda import LambdaEvent, LambdaProcedureMap, LambdaResponse

from pydantic_core import ValidationError
import logging

logger = logging.getLogger()
procedure_map = LambdaProcedureMap()

def lambda_handler(event, context) -> dict:
    try:
        event = LambdaEvent(**event)
    
    except ValidationError as e:
        logger.error(e)

        return LambdaResponse(
            status_code=400,
            body=str(e),
        ).model_dump()

    procedure_to_run = procedure_map[event.procedure]
    response = procedure_to_run(event.arguments)

    return response.model_dump()
