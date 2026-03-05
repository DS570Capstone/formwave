#!/bin/bash

echo "--------------------------------------"
echo "Creating database dump..."
echo "--------------------------------------"

mkdir -p backups

docker exec formwave-postgres \
pg_dump -U formwave -Fc formwave > backups/formwave_latest.dump

echo ""
echo "--------------------------------------"
echo "Backup completed successfully"
echo "--------------------------------------"
echo ""
echo "Dump file created:"
echo ""
echo "   backups/formwave_latest.dump"
echo ""
echo "Next step:"
echo ""
echo "Please upload this file manually to Google Drive:"
echo ""
echo "   https://drive.google.com/drive/folders/YOUR_FOLDER_LINK_HERE"
echo ""
echo "Upload file:"
echo ""
echo "   formwave_latest.dump"
echo ""
echo "--------------------------------------"