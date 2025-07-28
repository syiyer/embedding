Class Utils.Query Extends %RegisteredObject
{

/// Prompt for a query, embed it, and show the 3 most similar rows
ClassMethod FindTop3() As %Status
{
    set tSC = $$$OK
    Try {
                /* -------- 1. Ask for table name */
        Write !,"Enter the name of the pdf to search (e.g., 'UOB'): ", !
        Use $Principal::"S"
        Read tableName
        Use $Principal
        Set tableName = "Embedding."_tableName
        If $ZStrip(tableName,"<>W") = "" {
            Write !,"(empty table name – nothing to do)", !
        }

        /* -------------------------------------------------- 1. get user query */
        Write !,"Enter your search question: ", !
        Use $Principal::"S"                              ; force standard input
        Read queryLine 
        Use $Principal

        If $ZStrip(queryLine,"<>W")="" {
            Write !,"(empty query – nothing to do)", !
            // Quit $$$OK
        }

        /* -------------------------------------------------- 2. create embedding */
        // Set configName = "sentence-transformers-config"               ; change if you used another name
        //  Set queryVector = ##class(%Embedding.Config).Embedding(configName, queryLine)

        /* -------------------------------------------------- 3. run vector search */
        Set sql = "SELECT TOP 3 Description, Name, VECTOR_DOT_PRODUCT(DescriptionEmbedding, EMBEDDING(?, 'bge-base-config')) AS similarity "_
                "FROM "_tableName_" "_
                "ORDER BY similarity DESC"
        
        Set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(sql))
        Set rset = tStatement.%Execute(queryLine)

        /* -------------------------------------------------- 4. display results */
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE,rset.%Message)
        }
        set retrievedInfo = ""
        Write !!,"Top 3 most similar sections:", !!
        while rset.%Next() {
            Write "Score       : ", $FN(rset.similarity,"",3), !
            Write "Section     : ", rset.%Get("Name"), !
            Write "Description : ", rset.%Get("Description"), !!
        }
        // w "", !
        // w retrievedInfo
        // return retrievedInfo
        return $$$OK
    }
    catch e {
        set sc = e.AsStatus()
        return sc
    }
}

}
