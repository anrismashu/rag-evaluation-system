from pydantic import Field,BaseModel

class Answer(BaseModel):
    answer:str = Field(description="the answer that is needed for the users question")
    context:str = Field(description="the context agent used to replay to the user question and generate answer with help of this ")