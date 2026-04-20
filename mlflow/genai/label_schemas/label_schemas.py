from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mlflow.genai.utils.enum_utils import StrEnum

if TYPE_CHECKING:
    from databricks.agents.review_app import label_schemas as _label_schemas

    _InputCategorical = _label_schemas.InputCategorical
    _InputCategoricalList = _label_schemas.InputCategoricalList
    _InputNumeric = _label_schemas.InputNumeric
    _InputText = _label_schemas.InputText
    _InputTextList = _label_schemas.InputTextList
    _LabelSchema = _label_schemas.LabelSchema

DatabricksInputType = TypeVar("DatabricksInputType")
_InputType = TypeVar("_InputType", bound="InputType")


@dataclass
class CategoricalOption:
    """
    An option for a categorical label schema, with an optional description.

    Args:
        value: The option value (what gets recorded as the assessment).
        description: Optional human-readable description shown to labelers
            explaining what this option means.
    """

    value: str
    description: str | None = None


class InputType(ABC):
    """Base class for all input types."""

    @abstractmethod
    def _to_databricks_input(self) -> DatabricksInputType:
        """Convert to the internal Databricks input type."""

    @classmethod
    @abstractmethod
    def _from_databricks_input(cls, input_obj: DatabricksInputType) -> _InputType:
        """Create from the internal Databricks input type."""


@dataclass
class InputCategorical(InputType):
    """
    A single-select dropdown for collecting assessments from stakeholders.

    Options can be plain strings or :class:`CategoricalOption` instances with descriptions::

        # Without descriptions (existing behavior):
        InputCategorical(options=["Yes", "No"])

        # With descriptions:
        InputCategorical(options=[
            CategoricalOption("Correct", "Response fully answers the question"),
            CategoricalOption("Partial", "Response addresses the question but has gaps"),
            CategoricalOption("Incorrect", "Response is wrong or irrelevant"),
        ])

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    options: list[str | CategoricalOption]
    """List of available options for the categorical selection."""

    def _to_databricks_input(self) -> "_InputCategorical":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        plain_options = [
            o.value if isinstance(o, CategoricalOption) else o for o in self.options
        ]
        descriptions = {
            o.value: o.description
            for o in self.options
            if isinstance(o, CategoricalOption) and o.description is not None
        }
        return _label_schemas.InputCategorical(
            options=plain_options,
            option_descriptions=descriptions or None,
        )

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputCategorical") -> "InputCategorical":
        """Create from the internal Databricks input type."""
        descriptions = getattr(input_obj, "option_descriptions", None) or {}
        if descriptions:
            options: list[str | CategoricalOption] = [
                CategoricalOption(value=o, description=descriptions.get(o))
                for o in input_obj.options
            ]
        else:
            options = input_obj.options
        return cls(options=options)


@dataclass
class InputCategoricalList(InputType):
    """
    A multi-select dropdown for collecting assessments from stakeholders.

    Options can be plain strings or :class:`CategoricalOption` instances with descriptions::

        # Without descriptions (existing behavior):
        InputCategoricalList(options=["Tag A", "Tag B", "Tag C"])

        # With descriptions:
        InputCategoricalList(options=[
            CategoricalOption("PII", "Contains personally identifiable information"),
            CategoricalOption("Harmful", "Contains harmful or dangerous content"),
            CategoricalOption("Off-topic", "Does not address the user's question"),
        ])

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    options: list[str | CategoricalOption]
    """List of available options for the multi-select categorical (dropdown)."""

    def _to_databricks_input(self) -> "_InputCategoricalList":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        plain_options = [
            o.value if isinstance(o, CategoricalOption) else o for o in self.options
        ]
        descriptions = {
            o.value: o.description
            for o in self.options
            if isinstance(o, CategoricalOption) and o.description is not None
        }
        return _label_schemas.InputCategoricalList(
            options=plain_options,
            option_descriptions=descriptions or None,
        )

    @classmethod
    def _from_databricks_input(
        cls, input_obj: "_InputCategoricalList"
    ) -> "InputCategoricalList":
        """Create from the internal Databricks input type."""
        descriptions = getattr(input_obj, "option_descriptions", None) or {}
        if descriptions:
            options: list[str | CategoricalOption] = [
                CategoricalOption(value=o, description=descriptions.get(o))
                for o in input_obj.options
            ]
        else:
            options = input_obj.options
        return cls(options=options)


@dataclass
class InputTextList(InputType):
    """Like `Text`, but allows multiple entries.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    max_length_each: int | None = None
    """Maximum character length for each individual text entry. None means no limit."""

    max_count: int | None = None
    """Maximum number of text entries allowed. None means no limit."""

    def _to_databricks_input(self) -> "_InputTextList":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputTextList(
            max_length_each=self.max_length_each, max_count=self.max_count
        )

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputTextList") -> "InputTextList":
        """Create from the internal Databricks input type."""
        return cls(max_length_each=input_obj.max_length_each, max_count=input_obj.max_count)


@dataclass
class InputText(InputType):
    """A free-form text box for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    max_length: int | None = None
    """Maximum character length for the text input. None means no limit."""

    def _to_databricks_input(self) -> "_InputText":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputText(max_length=self.max_length)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputText") -> "InputText":
        """Create from the internal Databricks input type."""
        return cls(max_length=input_obj.max_length)


@dataclass
class InputNumeric(InputType):
    """A numeric input for collecting assessments from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    min_value: float | None = None
    """Minimum allowed numeric value. None means no minimum limit."""

    max_value: float | None = None
    """Maximum allowed numeric value. None means no maximum limit."""

    def _to_databricks_input(self) -> "_InputNumeric":
        """Convert to the internal Databricks input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        return _label_schemas.InputNumeric(min_value=self.min_value, max_value=self.max_value)

    @classmethod
    def _from_databricks_input(cls, input_obj: "_InputNumeric") -> "InputNumeric":
        """Create from the internal Databricks input type."""
        return cls(min_value=input_obj.min_value, max_value=input_obj.max_value)


class LabelSchemaType(StrEnum):
    """Type of label schema."""

    FEEDBACK = "feedback"
    EXPECTATION = "expectation"


@dataclass(frozen=True)
class LabelSchema:
    """A label schema for collecting input from stakeholders.

    .. note::
        This functionality is only available in Databricks. Please run
        `pip install mlflow[databricks]` to use it.
    """

    name: str
    """Unique name identifier for the label schema."""

    type: LabelSchemaType
    """Type of the label schema, either 'feedback' or 'expectation'."""

    title: str
    """Display title shown to stakeholders in the labeling review UI."""

    input: InputCategorical | InputCategoricalList | InputText | InputTextList | InputNumeric
    """
    Input type specification that defines how stakeholders will provide their assessment
    (e.g., dropdown, text box, numeric input)
    """
    instruction: str | None = None
    """Optional detailed instructions shown to stakeholders for guidance."""

    enable_comment: bool = False
    """Whether to enable additional comment functionality for reviewers."""

    @classmethod
    def _convert_databricks_input(cls, input_obj):
        """Convert a Databricks input type to the corresponding MLflow input type."""
        from databricks.agents.review_app import label_schemas as _label_schemas

        input_type_mapping = {
            _label_schemas.InputCategorical: InputCategorical,
            _label_schemas.InputCategoricalList: InputCategoricalList,
            _label_schemas.InputText: InputText,
            _label_schemas.InputTextList: InputTextList,
            _label_schemas.InputNumeric: InputNumeric,
        }

        input_class = input_type_mapping.get(type(input_obj))
        if input_class is None:
            raise ValueError(f"Unknown input type: {type(input_obj)}")

        return input_class._from_databricks_input(input_obj)

    @classmethod
    def _from_databricks_label_schema(cls, schema: "_LabelSchema") -> "LabelSchema":
        """Convert from the internal Databricks label schema type."""

        return cls(
            name=schema.name,
            type=schema.type,
            title=schema.title,
            input=cls._convert_databricks_input(schema.input),
            instruction=schema.instruction,
            enable_comment=schema.enable_comment,
        )
