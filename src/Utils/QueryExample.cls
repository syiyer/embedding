Class Utils.QueryExample Extends %RegisteredObject
{

/// Prompt for a query, embed it, and show the 3 most similar rows
ClassMethod FindTop3() As %Status
{
    set tSC = $$$OK
    Try {
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
        Set sql = "SELECT TOP 3 Description, Name FROM Embedding.HSBC " _
                  "ORDER BY VECTOR_DOT_PRODUCT(DescriptionEmbedding, EMBEDDING(?, 'jina-config')) DESC" 
                  

        Set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(sql))
        Set rset = tStatement.%Execute(queryLine)

        /* -------------------------------------------------- 4. display results */
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE,rset.%Message)
        }
        set retrievedInfo = ""
        while rset.%Next() {
            set retrievedInfo = retrievedInfo_$char(13,10)
            set retrievedInfo = retrievedInfo_"Section: "_rset.%Get("Name")
            set retrievedInfo = retrievedInfo_$Char(13,10)
            set retrievedInfo = retrievedInfo_"Description: "_rset.%Get("Description")
        }
        w "", !
        w retrievedInfo
        return retrievedInfo
    }
    catch e {
        set sc = e.AsStatus()
        return sc
    }
}

}
