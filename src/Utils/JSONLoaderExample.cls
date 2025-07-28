Class Utils.JSONLoaderExample Extends %RegisteredObject
{

ClassMethod LoadChunks(pFile As %String = "") As %Status
{
    Set tSC = $$$OK
    Write ">>> LoadChunks started", !
    Write ">>> File path: ", pFile, !

    Try {
        /* ---------- read file ---------- */
        Set stream = ##class(%Stream.FileCharacter).%New()
        Set sc = stream.LinkToFile(pFile)
        If $$$ISERR(sc) {
            Write "❌ Failed to open file", !
            Return sc
        }

        If stream.AtEnd {
            Write "❌ File is empty", !
            Return $System.Status.Error(5001,"Empty or unreadable file")
        }

        Set jsonString = stream.Read(stream.Size)
        Write "✔ File read successfully. Length: ", $Length(jsonString), !

        /* ---------- parse JSON ---------- */
        Set chunks = {}.%FromJSON(jsonString)
        If 'chunks.%Extends("%DynamicArray") {
            Write "❌ JSON is not an array", !
            Return $System.Status.Error(5001,"Top-level JSON must be an array")
        }

        Write "✔ JSON parsed. Row count: ", chunks.%Size(), !

        /* ---------- prepare statement ---------- */
        set myquery = "INSERT INTO Embedding.HSBC (Description,Name) VALUES (?,?)"
        Set stmt = ##class(%SQL.Statement).%New()
        Set sc = stmt.%Prepare(myquery)
        If $$$ISERR(sc) {
            Write "❌ Failed to prepare SQL statement", !
            Do $System.Status.DisplayError(sc)
            Return sc
        }
        Set inserted = 0
        /* ---------- iterate & insert ---------- */
        set iterator = chunks.%GetIterator()
        while iterator.%GetNext(.key, .value) {
            Set text = value.text
            // w text
            Set heading = value.heading
            // w heading
            If text = "" {
                Write "⚠ Skipping empty row #", key, !
                Continue
            }
            Set text    = $Extract(text,1,512)
            Set heading = $Extract(heading,1,128)

            Try {
                Do stmt.%Execute(text, heading)
                Write "✅ Inserted row #", key, ": ", $Extract(heading,1,30), !
                Set inserted = inserted + 1
            } Catch ex {
                Write "❌ Insert failed at row #", key, ": ", ex.DisplayString(), !
            }
        }

        Write "🎯 Done. Inserted ", inserted, " rows.", !
    }
    Catch ex {
        Set tSC = ex.AsStatus()
        Write "❌ General exception: ", ex.DisplayString(), !
    }

    Return tSC
}

}
