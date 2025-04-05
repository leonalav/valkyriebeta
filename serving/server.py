from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio
from typing import List, Dict

app = FastAPI()
