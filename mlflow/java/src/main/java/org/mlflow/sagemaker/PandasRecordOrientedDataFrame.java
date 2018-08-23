package org.mlflow.sagemaker;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import org.mlflow.utils.SerializationUtils;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import com.fasterxml.jackson.core.JsonProcessingException;

class PandasRecordOrientedDataFrame {
  private final List<Map<String, Object>> records;

  private static final String LEAP_FRAME_KEY_ROWS = "rows";
  private static final String LEAP_FRAME_KEY_SCHEMA = "schema";

  PandasRecordOrientedDataFrame(List<Map<String, Object>> records) {
    this.records = records;
  }

  static PandasRecordOrientedDataFrame fromJson(String frameJson) throws IOException {
    return new PandasRecordOrientedDataFrame(SerializationUtils.fromJson(frameJson, List.class));
  }

  /**
   * @return The number of records contained in the dataframe
   */
  public int size() {
    return this.records.size();
  }

  /**
   * Applies the specified MLeap frame schema ({@link LeapFrameSchema}) to this dataframe,
   * producing a {@link DefaultLeapFrame}
   *
   * @throws MissingSchemaFieldException If the supplied pandas dataframe is missing a required
   *     field
   */
  DefaultLeapFrame toLeapFrame(LeapFrameSchema leapFrameSchema) throws JsonProcessingException {
    List<List<Object>> mleapRows = new ArrayList<>();
    for (Map<String, Object> record : this.records) {
      List<Object> mleapRow = new ArrayList<>();
      for (String fieldName : leapFrameSchema.getFieldNames()) {
        if (!record.containsKey(fieldName)) {
          throw new MissingSchemaFieldException(fieldName);
        }
        mleapRow.add(record.get(fieldName));
      }
      mleapRows.add(mleapRow);
    }
    Map<String, Object> rawFrame = new HashMap<>();
    rawFrame.put(LEAP_FRAME_KEY_ROWS, mleapRows);
    rawFrame.put(LEAP_FRAME_KEY_SCHEMA, leapFrameSchema.getRawSchema());
    String leapFrameJson = SerializationUtils.toJson(rawFrame);
    DefaultLeapFrame leapFrame = LeapFrameUtils.getLeapFrameFromJson(leapFrameJson);
    return leapFrame;
  }
}
