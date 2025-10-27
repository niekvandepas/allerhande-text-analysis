IssueContents = dict[str, str]  # page_number -> OCR text
Issues = dict[str, IssueContents]  # issue date -> issue contents
TextsByYear = dict[int, Issues]  # year -> issues
TextsByTimeSlice = dict[int, Issues]  # decade -> issues
