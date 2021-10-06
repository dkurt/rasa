from __future__ import annotations
import json
from json.decoder import JSONDecodeError
import logging
import re
from typing import Any, Dict, Match, Optional, Pattern, Text, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.constants import DOCS_URL_STORIES, INTENT_MESSAGE_PREFIX
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class RegexMessageHandlerGraphComponent(GraphComponent):
    """Unpacks messages where `TEXT` contains an encoding of attributes.

    The `TEXT` attribute of such messages consists of the following sub-strings:
    1. special symbol "/" (mandatory)
    2. intent name (mandatory)
    3. "@<confidence value>" where the value can be any int or float (optional)
    4. string representation of a dictionary mapping entity types to entity
       values (optional)
    """

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexMessageHandlerGraphComponent:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls()

    # TODO: Handle empty domain (NLU only training)
    def process(
        self, messages: List[Message], domain: Optional[Domain] = None
    ) -> List[Message]:
        """Unpacks messages where `TEXT` contains an encoding of attributes.

        Note that this method returns a *new* message instance if there is
        something to unpack in the given message (and returns the given message
        otherwise). The new message is created on purpose to get rid of all attributes
        that NLU components might have added based on the `TEXT` attribute which
        does not contain real text but the regex we expect here.

        Args:
            messages: list of messages
            domain: the domain
        Returns:
            list of messages where the i-th message is equal to the i-th input message
            if that message does not need to be unpacked, and a new message with the
            extracted attributes otherwise
        """
        return [
            YAMLStoryReader.unpack_regex_message(message, domain)
            for message in messages
        ]
