package com.databricks.mlflow.mleap;

import com.databricks.mlflow.utils.FileUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;

@JsonIgnoreProperties(ignoreUnknown = true)
public class MLeapTransformerSchema {
    private static final String LEAP_FRAME_KEY_ROWS = "rows";
    private static final String LEAP_FRAME_KEY_SCHEMA = "schema";

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class SchemaField {
        @JsonProperty("name") private String name;
    }

    @JsonProperty("fields") private List<SchemaField> fields;

    private String text;

    public static MLeapTransformerSchema fromFile(String filePath) throws IOException {
        MLeapTransformerSchema newSchema =
            FileUtils.parseJsonFromFile(filePath, MLeapTransformerSchema.class);
        String schemaText = new String(Files.readAllBytes(Paths.get(filePath)));
        newSchema.setText(schemaText);
        return newSchema;
    }

    public List<String> getOrderedFieldNames() {
        List<String> orderedFieldNames = new ArrayList<>();
        for (SchemaField field : fields) {
            orderedFieldNames.add(field.name);
        }
        return orderedFieldNames;
    }

    private void setText(String text) {
        this.text = text;
    }

    /**
     * Converts a Pandas dataframe JSON in `record` format to MLeap frame JSON in
     * `row` format using the schema defined by this schema object
     */
    public String applyToPandasRecordJson(String pandasJson) throws IOException {
        List<String> orderedFieldNames = getOrderedFieldNames();
        ObjectMapper mapper = new ObjectMapper();
        List<Map<String, Object>> pandasRecords =
            mapper.readValue(pandasJson, new TypeReference<List<Map<String, Object>>>() {});
        List<List<Object>> mleapRows = new ArrayList<>();
        for (Map<String, Object> record : pandasRecords) {
            List<Object> mleapRow = new ArrayList<>();
            for (String fieldName : orderedFieldNames) {
                mleapRow.add(record.get(fieldName));
            }
            mleapRows.add(mleapRow);
        }
        String serializedRows = mapper.writeValueAsString(mleapRows);
        String leapFrameJson = String.format("{ \"%s\" : %s, \"%s\" : %s }", LEAP_FRAME_KEY_ROWS,
            serializedRows, LEAP_FRAME_KEY_SCHEMA, this.text);
        return leapFrameJson;
    }
}
