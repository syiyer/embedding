Class Utils.JSONLoader Extends %RegisteredObject
{

ClassMethod LoadChunks(pdfSlug As %String, jsonPath As %String) As %Integer
{
    Try {
            
        Set tableName = "Embedding.Financial"
        Set filePath  = "C:\Users\Siyer\OneDrive - InterSystems Corporation\Documents\embedding\data\"_pdfSlug_"\chunks.json"
        If '##class(%File).Exists(filePath) {
            Return 0
        }
        
        Set createSQL = "CREATE TABLE IF NOT EXISTS "_tableName_" ("_
            "Name VARCHAR(200), "_
            "Length INT,"_
            "Description LONGVARCHAR, "_
            "DescriptionEmbedding EMBEDDING('bge-base-config','Description'),"_  
            "NameEmbedding EMBEDDING('bge-base-config','Name'),"_
            "PDF VARCHAR(50)"_
            ")" 
        Set stmtCreate = ##class(%SQL.Statement).%New()
        do stmtCreate.%Prepare(createSQL)
        Set scTable = stmtCreate.%Execute()
        set idxExist = 0
        set exists = "SELECT COUNT(*) as cnt "_ 
            "FROM INFORMATION_SCHEMA.INDEXES "_
           "WHERE TABLE_SCHEMA = 'Embedding' "_
             "AND TABLE_NAME   = 'Financial' "_
             "AND INDEX_NAME   = 'HNSWIndex'"
        set stmtIdx = ##class(%SQL.Statement).%New()
        do stmtIdx.%Prepare(exists)
        Set rs = stmtIdx.%Execute()
        If rs.%Next() set idxExist = rs.%Get("cnt")
        If idxExist = 0 {
            set nextsql = "CREATE INDEX HNSWIndex"_ 
                "ON TABLE "_tableName_" (DescriptionEmbedding)"_
                "AS HNSW(M=32, efConstruction=100, Distance='DotProduct')"
            set nextIdx = ##class(%SQL.Statement).%New()
            do nextIdx.%Prepare(nextsql)
            Set scIndex = nextIdx.%Execute()
            if $$$ISERR(scIndex) {
                Return scIndex
            }
        }
        
        Set stream = ##class(%Stream.FileCharacter).%New()
        Set scStream = stream.LinkToFile(filePath)
        If $$$ISERR(scStream) {
            Return scStream
        }

        If stream.AtEnd {
            Return $System.Status.Error(5001,"Empty or unreadable file")
        }

        Set jsonString = stream.Read(stream.Size)

        
        Set chunks = {}.%FromJSON(jsonString)
        If 'chunks.%Extends("%DynamicArray") {
            Return $System.Status.Error(5001,"Top-level JSON must be an array")
        }

        set myquery = "INSERT INTO "_tableName_" (Description, Length, Name, PDF) VALUES (?, ?, ?, ?)"
        Set stmt = ##class(%SQL.Statement).%New()
        Set scInsert = stmt.%Prepare(myquery)
        If $$$ISERR(scInsert) {
            Return scInsert
        }
        Set inserted = 0
        
        set iterator = chunks.%GetIterator()
        while iterator.%GetNext(.key, .value) {
            Set text = value.text
            
            Set heading = value.heading
            
            Set length = value.tokens
            If text = "" {
                Continue
            }
            Set text    = $Extract(text,1,1024)
            Set heading = $Extract(heading,1,200)
            set length = $Extract(length,1,5)


            Try {
                Do stmt.%Execute(text, length, heading, pdfSlug)
                Set inserted = inserted + 1
            } Catch ex {
            }
        }
    }
    Catch ex {
        Set tSC = ex.AsStatus()
    }

    Return inserted
}

}
